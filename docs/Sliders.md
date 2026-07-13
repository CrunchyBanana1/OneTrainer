# Krea2 Concept Sliders (local tool)

Hardcoded image/text sliders for Krea2 LoRA training. Configure in
`modules/trainer/slider_config.py`, then run a normal Krea2 LoRA training.
Set `SLIDER_MODE = None` to disable (default).

## Layer targeting recommendation ("leave text alone")
For cleaner sliders, target the LoRA at visual layers (self-attention, conv,
time-embedding) and exclude text-conditioning layers, via your LoRA layer
selection in the training config.

## Image slider
- Set `SLIDER_MODE = "image"`. Run at **batch_size = 1** (the slider applies one
  +1/-1 sign per batch; it raises if batch size > 1).
- Dataset: one concept whose `path` is the **parent** folder, containing parallel
  subfolders whose paths contain `IMAGE_POSITIVE_TOKEN` / `IMAGE_NEGATIVE_TOKEN`
  (default `positive`/`negative`), holding same-named image pairs. Captions can
  match or be empty.
- **REQUIRED: enable the concept's "Include Subdirectories" switch.** It defaults
  to OFF, and the images live in subfolders — with it off OneTrainer scans only
  the (empty) parent folder, finds no images, and never trains. The concept
  preview image count is a quick check: 0 means the switch is still off.
- **Pairing is mandatory.** Each `positive/` image must have a same-named twin in
  `negative/` (matched by path + identical latent shape). At epoch start the batches
  are reordered so each positive trains immediately before its negative twin;
  positives with no matching twin are **dropped** (logged as `[image slider] ...
  dropped N`). This contrastive pairing is what lets the LoRA learn only the concept
  direction — without it, training diverges (loss climbs).
- **Set gradient accumulation = 2** (or any multiple of 2). With twins adjacent, an
  accumulation of 2 folds each positive+negative pair into a single optimizer step,
  so the shared image content cancels exactly and only the concept direction is
  learned. This is the recommended setting.
- If loss still climbs, lower the learning rate (slider LoRAs typically want it
  lower than a normal LoRA).

## Text slider (dataset mode)
- Set `SLIDER_MODE = "text"`. Provide any Krea2 dataset for starting latents.
- Set `TEXT_POSITIVE`, `TEXT_NEUTRAL`, `TEXT_NEGATIVE`, `TEXT_TARGET`,
  `TEXT_GUIDANCE`. `TEXT_TARGET` is usually identical to `TEXT_NEUTRAL`.
- Loss = MSE(student, neutral + guidance*(positive - negative)).

## Known limitations
- Plain LoRA only (DoRA/LoKr multiplier unsupported).
- Quantized transformer: the multiplier is not applied on the
  `forward_with_lora` path (`LoRAModule.py`); use a non-quantized run.
