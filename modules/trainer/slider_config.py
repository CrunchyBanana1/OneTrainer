"""
Hardcoded configuration for the Krea2 text concept slider.

Edit the constants below and set SLIDER_MODE = "text" to enable slider training.
When SLIDER_MODE is None, normal training is completely unaffected.

This is intentionally NOT wired into the UI — it is a local training tool.
"""
import math

# None | "text"
SLIDER_MODE: str | None = "text"

# ---- text slider (dataset mode) --------------------------------------------
# The attribute direction, as prompts. TEXT_TARGET is what the student is
# conditioned on (typically identical to TEXT_NEUTRAL).
TEXT_POSITIVE: str = "a woman wearing a short cropped top exposing underboob"
TEXT_NEUTRAL: str = "a woman wearing a top"
TEXT_NEGATIVE: str = "a woman wearing a long draped top"
TEXT_TARGET: str = "a woman"
TEXT_GUIDANCE: float = 2.0


def validate_slider_config() -> None:
    """Fail fast with a readable message when the active mode is misconfigured."""
    if SLIDER_MODE is None:
        return
    if SLIDER_MODE == "text":
        for name in ("TEXT_POSITIVE", "TEXT_NEUTRAL", "TEXT_NEGATIVE", "TEXT_TARGET"):
            if not globals()[name]:
                raise ValueError(f"Text slider: {name} must be a non-empty prompt.")
        if not math.isfinite(TEXT_GUIDANCE):
            raise ValueError("Text slider: TEXT_GUIDANCE must be a finite number.")
    else:
        raise ValueError(f"Unknown SLIDER_MODE {SLIDER_MODE!r}; expected None or 'text'.")
