"""
Krea2 concept slider training steps (image + text), driven by slider_config.

Enabled via slider_config.SLIDER_MODE; dispatched from GenericTrainer. Krea2 LoRA only.
"""
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
