"""
Hardcoded configuration for Krea2 concept sliders.

Edit the constants below and set SLIDER_MODE to enable slider training.
When SLIDER_MODE is None, normal training is completely unaffected.

This is intentionally NOT wired into the UI — it is a local training tool.
"""
import math

# None | "image" | "text"
SLIDER_MODE: str = None # | None = None

#SLIDER_MODE = "image"

# ---- image slider ----------------------------------------------------------
# Dataset layout: one concept with parallel subfolders whose paths contain
# these tokens, holding same-named image pairs. Run at batch_size = 1.
#   .../positive/foo.png   and   .../negative/foo.png
IMAGE_POSITIVE_TOKEN: str = "positive"
IMAGE_NEGATIVE_TOKEN: str = "negative"

# ---- text slider (dataset mode) --------------------------------------------
# The attribute direction, as prompts. TEXT_TARGET is what the student is
# conditioned on (typically identical to TEXT_NEUTRAL).
TEXT_POSITIVE: str = "a person with a wide smile"
TEXT_NEUTRAL: str = "a person"
TEXT_NEGATIVE: str = "a person with a neutral expression"
TEXT_TARGET: str = "a person"
TEXT_GUIDANCE: float = 2.0

# When True, the image slider prints one line per step (filename, +1/-1 sign, shared pair seed)
# so you can verify each positive and its negative twin pair up with the same seed and opposite
# sign. Leave False for normal runs.
SLIDER_DEBUG: bool = False


def validate_slider_config() -> None:
    """Fail fast with a readable message when the active mode is misconfigured."""
    if SLIDER_MODE is None:
        return
    if SLIDER_MODE == "image":
        if not IMAGE_POSITIVE_TOKEN or not IMAGE_NEGATIVE_TOKEN:
            raise ValueError("Image slider: IMAGE_POSITIVE_TOKEN and IMAGE_NEGATIVE_TOKEN must be non-empty.")
    elif SLIDER_MODE == "text":
        for name in ("TEXT_POSITIVE", "TEXT_NEUTRAL", "TEXT_NEGATIVE", "TEXT_TARGET"):
            if not globals()[name]:
                raise ValueError(f"Text slider: {name} must be a non-empty prompt.")
        if not math.isfinite(TEXT_GUIDANCE):
            raise ValueError("Text slider: TEXT_GUIDANCE must be a finite number.")
    else:
        raise ValueError(f"Unknown SLIDER_MODE {SLIDER_MODE!r}; expected None, 'image', or 'text'.")
