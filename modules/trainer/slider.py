"""
Krea2 text concept slider training, driven by slider_config.

Enabled via slider_config.SLIDER_MODE == "text"; dispatched from GenericTrainer. Krea2 LoRA only.

A frozen teacher (LoRA multiplier 0) builds a guided target from positive/neutral/negative
prompts at a shared noised latent, and the student (LoRA multiplier 1) is trained to match it:
    target = neutral + guidance * (positive - negative)
    loss   = MSE(student, target)
"""
from dataclasses import dataclass

from modules.module.quantized.LinearSVD import BaseLinearSVD
from modules.trainer import slider_config

import torch
import torch.nn.functional as F
from torch import Tensor


def compose_text_target(neutral: Tensor, positive: Tensor, negative: Tensor, guidance: float) -> Tensor:
    """Concept-slider guided target: neutral + guidance * (positive - negative)."""
    return neutral + guidance * (positive - negative)


_PROMPT_CACHE = None


def reset_prompt_cache() -> None:
    """Clear the memoized slider prompt encodings.

    OneTrainer's UI reuses one process across successive train() calls, so the module-global
    cache would otherwise leak a previous run's cached embeds (from a different model) into a
    new run. Call this at the start of every training run that has the slider enabled.
    """
    global _PROMPT_CACHE
    _PROMPT_CACHE = None


def check_slider_compatible(model) -> None:
    """Raise fast if `model` cannot correctly support slider training.

    Slider training relies on `model.transformer_lora`'s multiplier to scale the LoRA delta for
    the teacher (0) and student (1) passes. That multiplier is only honored on the plain
    `LoRAModule.delta_forward` path. If the transformer's linears are quantized, LoRA wraps
    `BaseLinearSVD` and `LoRAModule.forward` takes the `forward_with_lora(...)` path instead,
    which ignores `self.multiplier` entirely -- silently breaking the slider (the teacher pass
    would run with the LoRA active instead of frozen).
    """
    transformer_lora = getattr(model, "transformer_lora", None)
    if transformer_lora is None:
        raise RuntimeError(
            "Slider training requires a Krea2 LoRA model (model.transformer_lora is None)."
        )
    for module in transformer_lora.lora_modules.values():
        # Read the underlying `_orig_module` attribute directly instead of the `orig_module`
        # property: `PeftBase.orig_module` asserts `self._orig_module is not None` and raises
        # AssertionError (not AttributeError) for dummy modules (orig_module=None at
        # construction), so getattr(..., "orig_module", None) would not suppress it.
        # `_orig_module` is `[orig_module]` or `None` (see PeftBase.__init__); modules without
        # a `_orig_module` attribute at all (e.g. FusedModuleGroup) are also skipped.
        orig_list = getattr(module, "_orig_module", None)
        if not orig_list:
            continue
        if isinstance(orig_list[0], BaseLinearSVD):
            raise RuntimeError(
                "Slider training does not support a quantized transformer: the LoRA multiplier "
                "is ignored on the quantized (BaseLinearSVD) forward path, which would silently "
                "produce a wrong slider. Train with a non-quantized transformer."
            )


@dataclass
class SliderPromptCache:
    positive: tuple           # (text_encoder_output, mask)
    neutral: tuple
    negative: tuple
    target: tuple


def get_prompt_cache(model, train_device) -> SliderPromptCache:
    """Encode the fixed slider prompts once and memoize them."""
    global _PROMPT_CACHE
    if _PROMPT_CACHE is None:
        def enc(text):
            return model.encode_text(train_device=train_device, batch_size=1, text=text)
        _PROMPT_CACHE = SliderPromptCache(
            positive=enc(slider_config.TEXT_POSITIVE),
            neutral=enc(slider_config.TEXT_NEUTRAL),
            negative=enc(slider_config.TEXT_NEGATIVE),
            target=enc(slider_config.TEXT_TARGET),
        )
    return _PROMPT_CACHE


def text_slider_step(model_setup, model, batch, config, train_progress):
    cache = get_prompt_cache(model, model_setup.train_device)
    with model.autocast_context:
        batch_seed = train_progress.global_step
        generator = torch.Generator(device=config.train_device)
        generator.manual_seed(batch_seed)
        prep = model_setup._prepare_noised_latent(model, batch, config, generator, deterministic=False)
        latent_input = prep['scaled_noisy_latent_image']
        timestep = prep['timestep']

        with torch.no_grad():
            model.transformer_lora.set_multiplier(0.0)  # teacher: LoRA off
            pos = model_setup._transformer_forward(model, latent_input, cache.positive[0], cache.positive[1], timestep)
            neu = model_setup._transformer_forward(model, latent_input, cache.neutral[0], cache.neutral[1], timestep)
            neg = model_setup._transformer_forward(model, latent_input, cache.negative[0], cache.negative[1], timestep)
            target = compose_text_target(neu, pos, neg, slider_config.TEXT_GUIDANCE).detach()

        model.transformer_lora.set_multiplier(1.0)  # student: LoRA on
        student = model_setup._transformer_forward(model, latent_input, cache.target[0], cache.target[1], timestep)
        loss = F.mse_loss(student.float(), target.float())
    return loss


def slider_enabled() -> bool:
    return slider_config.SLIDER_MODE is not None


def slider_train_step(model_setup, model, batch, config, train_progress):
    if slider_config.SLIDER_MODE == "text":
        return text_slider_step(model_setup, model, batch, config, train_progress)
    raise ValueError(f"slider_train_step called with SLIDER_MODE={slider_config.SLIDER_MODE!r}")
