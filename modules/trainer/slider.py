"""
Krea2 concept slider training steps (image + text), driven by slider_config.

Enabled via slider_config.SLIDER_MODE; dispatched from GenericTrainer. Krea2 LoRA only.
"""
from dataclasses import dataclass

from modules.module.quantized.LinearSVD import BaseLinearSVD
from modules.trainer import slider_config

import torch
import torch.nn.functional as F
from torch import Tensor


def image_multiplier_for_path(image_path: str, positive_token: str, negative_token: str) -> float:
    """+1.0 for a positive-folder image, -1.0 for a negative-folder image."""
    has_pos = positive_token in image_path
    has_neg = negative_token in image_path
    if has_pos and has_neg:
        raise ValueError(f"Image path {image_path!r} contains both tokens; folder layout is ambiguous.")
    if has_pos:
        return 1.0
    if has_neg:
        return -1.0
    raise ValueError(f"Image path {image_path!r} contains neither {positive_token!r} nor {negative_token!r}.")


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
    the positive/negative/teacher/student passes. That multiplier is only honored on the plain
    `LoRAModule.delta_forward` path. If the transformer's linears are quantized, LoRA wraps
    `BaseLinearSVD` and `LoRAModule.forward` takes the `forward_with_lora(...)` path instead,
    which ignores `self.multiplier` entirely -- silently breaking both the image and text
    sliders (every sample would be trained as if multiplier=1, regardless of folder/token sign).
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


def image_slider_step(model_setup, model, batch, config, train_progress):
    if len(batch['image_path']) != 1:
        raise ValueError(
            f"image_slider_step got a batch of {len(batch['image_path'])} images, but the image "
            f"slider applies a single +1/-1 sign to the whole batch based on batch['image_path'][0]. "
            f"Set batch_size=1 for the image slider."
        )
    multiplier = image_multiplier_for_path(
        batch['image_path'][0], slider_config.IMAGE_POSITIVE_TOKEN, slider_config.IMAGE_NEGATIVE_TOKEN
    )
    model.transformer_lora.set_multiplier(multiplier)
    try:
        # Pair-aligned seed: build_image_slider_batches emits twins adjacently, so with batch_size=1
        # a positive lands on an even global_step and its negative twin on the next (odd) step.
        # global_step // 2 gives both the SAME seed -> identical noise + timestep, so only the concept
        # and the +1/-1 sign differ. Without this the twins are noised differently and the shared
        # content never cancels, so the LoRA drifts and the loss climbs. (Matches dxqb's Flux slider.)
        pair_seed = train_progress.global_step // 2
        data = model_setup.predict(model, batch, config, train_progress, seed_override=pair_seed)
        return model_setup.calculate_loss(model, batch, data, config)
    finally:
        # Restore the neutral multiplier so a following sample/preview doesn't inherit the +1/-1
        # slider sign left over from this training step.
        model.transformer_lora.set_multiplier(1.0)


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
            model.transformer_lora.set_multiplier(0.0)  # teacher: LoRA off
            neu = model_setup._transformer_forward(model, latent_input, cache.neutral[0], cache.neutral[1], timestep)
            model.transformer_lora.set_multiplier(0.0)  # teacher: LoRA off
            neg = model_setup._transformer_forward(model, latent_input, cache.negative[0], cache.negative[1], timestep)
            target = compose_text_target(neu, pos, neg, slider_config.TEXT_GUIDANCE).detach()

        model.transformer_lora.set_multiplier(1.0)  # student: LoRA on
        student = model_setup._transformer_forward(model, latent_input, cache.target[0], cache.target[1], timestep)
        loss = F.mse_loss(student.float(), target.float())
    return loss


def slider_enabled() -> bool:
    return slider_config.SLIDER_MODE is not None


def image_slider_enabled() -> bool:
    return slider_config.SLIDER_MODE == "image"


def build_image_slider_batches(batches, positive_token=None, negative_token=None):
    """Materialize one epoch of batches and reorder them into positive->negative twin pairs.

    The image slider is contrastive: each positive-folder image must be trained together with
    its negative twin (the same file under the negative folder) so the shared image content
    cancels and only the concept direction is learned. This materializes the whole epoch (fine
    for the small datasets sliders use), matches each positive with its negative twin -- found by
    substituting `positive_token` -> `negative_token` in the path and requiring an identical
    latent shape -- and emits the two adjacently (positive then negative). Positives without
    exactly one matching twin are dropped.

    Returns `(paired_batches, dropped_paths)`. Runs at batch_size=1 (one image per batch); raises
    otherwise. Set gradient accumulation = 2 so each optimizer step covers a whole pair, making
    the content-gradient cancellation exact.
    """
    if positive_token is None:
        positive_token = slider_config.IMAGE_POSITIVE_TOKEN
    if negative_token is None:
        negative_token = slider_config.IMAGE_NEGATIVE_TOKEN

    orig_list = list(batches)
    for batch in orig_list:
        if len(batch['image_path']) != 1:
            raise ValueError(
                f"The image slider requires batch_size=1 (one image per batch), got "
                f"{len(batch['image_path'])}."
            )

    paired = []
    dropped = []
    for batch in orig_list:
        path = batch['image_path'][0]
        if positive_token not in path:
            # negatives are pulled in via their positive twin; skip standalone negatives here
            continue
        negative_path = path.replace(positive_token, negative_token)
        twins = [
            item for item in orig_list
            if item['image_path'][0] == negative_path
            and batch['latent_image'][0].shape == item['latent_image'][0].shape
        ]
        if len(twins) == 1:
            paired.append(batch)
            paired.append(twins[0])
        else:
            dropped.append(path)
    return paired, dropped


def slider_train_step(model_setup, model, batch, config, train_progress):
    if slider_config.SLIDER_MODE == "image":
        return image_slider_step(model_setup, model, batch, config, train_progress)
    if slider_config.SLIDER_MODE == "text":
        return text_slider_step(model_setup, model, batch, config, train_progress)
    raise ValueError(f"slider_train_step called with SLIDER_MODE={slider_config.SLIDER_MODE!r}")
