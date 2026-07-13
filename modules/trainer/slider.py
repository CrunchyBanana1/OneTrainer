"""
Krea2 concept slider training steps (image + text), driven by slider_config.

Enabled via slider_config.SLIDER_MODE; dispatched from GenericTrainer. Krea2 LoRA only.
"""
from dataclasses import dataclass

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
    multiplier = image_multiplier_for_path(
        batch['image_path'][0], slider_config.IMAGE_POSITIVE_TOKEN, slider_config.IMAGE_NEGATIVE_TOKEN
    )
    model.transformer_lora.set_multiplier(multiplier)
    data = model_setup.predict(model, batch, config, train_progress)
    return model_setup.calculate_loss(model, batch, data, config)


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


def slider_train_step(model_setup, model, batch, config, train_progress):
    if slider_config.SLIDER_MODE == "image":
        return image_slider_step(model_setup, model, batch, config, train_progress)
    if slider_config.SLIDER_MODE == "text":
        return text_slider_step(model_setup, model, batch, config, train_progress)
    raise ValueError(f"slider_train_step called with SLIDER_MODE={slider_config.SLIDER_MODE!r}")
