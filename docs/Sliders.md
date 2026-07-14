# Krea2 Text Concept Slider (local tool)

A hardcoded text-driven concept slider for Krea2 LoRA training. Configure in
`modules/trainer/slider_config.py`, then run a normal Krea2 LoRA training.
Set `SLIDER_MODE = None` to disable (normal training is then completely unaffected).

## How it works
A frozen teacher (LoRA multiplier 0) runs the transformer on the batch's noised
latent with the positive / neutral / negative prompts and builds a guided target:

    target = neutral + guidance * (positive - negative)

The student (LoRA multiplier 1) runs once with the target prompt, and the loss is
`MSE(student, target)`. The teacher's three passes are under `torch.no_grad`, so
they're cheaper than they look. This is the standard concept-slider objective
(ostris / ai-toolkit), which trains stably.

## Setup
- Set `SLIDER_MODE = "text"`.
- Provide any Krea2 dataset — its latents are only used as the starting point to
  denoise from; the concept direction comes entirely from the prompts.
- Set `TEXT_POSITIVE`, `TEXT_NEUTRAL`, `TEXT_NEGATIVE`, `TEXT_TARGET`, `TEXT_GUIDANCE`.
  `TEXT_TARGET` is usually identical to `TEXT_NEUTRAL`.
- Gradient accumulation is **not** required — each step is a complete contrastive
  update on its own. Use whatever batch size / accumulation you'd normally use.

## Layer targeting recommendation ("leave text alone")
For cleaner sliders, target the LoRA at visual layers (self-attention, conv,
time-embedding) and exclude text-conditioning layers, via your LoRA layer
selection in the training config.

## Known limitations
- Plain LoRA only (DoRA/LoKr ignore the multiplier). Training fails fast with a
  clear error if the transformer is quantized (the multiplier is not honored on the
  `forward_with_lora` path), so use a non-quantized transformer.
