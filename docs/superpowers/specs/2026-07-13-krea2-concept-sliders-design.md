# Krea2 Concept Sliders — Design

**Date:** 2026-07-13
**Branch:** `krea2-concept-sliders` (based on `pyside6-dataset-tool-lmstudio-pr`, pushed to fork `CrunchyBanana1/OneTrainer` when ready)
**Status:** Approved design, pre-implementation
**Scope:** Local-only tooling. No upstream PR.

## Background

Two old OneTrainer forks by `dxqb` implemented concept sliders for **Flux.1-dev**:

- `dxqb/OneTrainer@text_slider` (core commit `851b12f`) — a teacher/student concept slider driven by text-prompt directions.
- `dxqb/OneTrainer@image_slider` (core commit `4e81383`) — a paired positive/negative image slider driven by a signed LoRA multiplier.

We are **porting the technique** to `krea2`. Note: `krea2` in this codebase is **not** Flux. `modules/model/Krea2Model.py` is built on a Qwen-image transformer (`Krea2Transformer2DModel`, `AutoencoderKLQwenImage`) with a `Qwen3VL` text encoder. The slider *technique* is architecture-agnostic, but every mechanical detail (text encoding, the transformer forward signature, the scheduler) is re-wired against the Krea2 setup classes rather than copied from the Flux ones.

## Decisions (locked)

| Decision | Choice |
|---|---|
| Which sliders | **Both.** Image slider first (foundation), then text slider. |
| Config UX | **Hardcoded**, like the original — constants in a dedicated module, edit Python to iterate. No UI widgets. |
| Text slider modes | **Dataset mode only.** No "generate" mode (skips the in-loop mini-diffusion). |
| Base architecture | Krea2 LoRA training. |

## Non-goals (YAGNI)

- No UI integration (no Base/Ctk/PySide6 view work).
- No "generate" mode for the text slider.
- No support for module types other than plain LoRA (DoRA / LoKr / etc. are out of scope — sliders are used with plain LoRA).
- No upstream PR; no attempt to keep GenericTrainer's slider path model-generic beyond Krea2.

## Architecture

Three pieces: a shared LoRA primitive, a hardcoded config + gate, and a slider train-step helper that GenericTrainer delegates to.

### 1. Shared primitive — LoRA multiplier

Add `set_multiplier(m: float)` to `LoRAModuleWrapper` in `modules/module/LoRAModule.py`, modeled on the existing `set_dropout` (`LoRAModule.py:1123`):

```python
def set_multiplier(self, multiplier: float):
    for module in self.lora_modules.values():
        module.multiplier = multiplier
```

Add a `multiplier: float = 1.0` attribute to the base peft module and factor it into the delta produced in each module's `forward` (i.e. the returned LoRA contribution is scaled by `multiplier`). For plain LoRA this is equivalent to scaling the effective `alpha`. Default `1.0` ⇒ **zero behavioral change** when sliders are unused.

Both sliders ride on this one capability:
- **Image slider:** `+1` on positive samples, `−1` on negative samples.
- **Text slider:** `0` for the frozen teacher passes (LoRA contributes nothing), `1` for the student pass.

### 2. Hardcoded config + gate

New module `modules/trainer/slider_config.py` holding all hardcoded parameters and the mode switch:

```python
SLIDER_MODE = None          # None | "image" | "text"

# image slider
IMAGE_POSITIVE_TOKEN = "positive"
IMAGE_NEGATIVE_TOKEN = "negative"

# text slider
TEXT_POSITIVE = "..."
TEXT_NEUTRAL  = "..."
TEXT_NEGATIVE = "..."
TEXT_TARGET   = "..."        # prompt the student is conditioned on (typically == neutral)
TEXT_GUIDANCE = 2.0
```

GenericTrainer gets a **thin guard** at its single train-step seam (`GenericTrainer.py:746`, the normal `predict → calculate_loss` path). When `SLIDER_MODE` is set, it delegates to the slider helper; when `None`, the existing code path runs **unchanged**. This keeps the 887-line shared file barely touched and makes normal training risk-free.

### 3. Slider train-step helper

New module `modules/trainer/slider.py` containing the two slider step functions. Each receives the same objects the normal step has (`model_setup`, `model`, `batch`, `config`, `train_progress`) and returns a `loss` tensor, so GenericTrainer's accumulation/backward/logging is reused verbatim.

## Component: Image slider

**Dataset layout:** a concept whose images live in two parallel subfolders — `.../positive/foo.png` and `.../negative/foo.png` — same filename in each. Captions matching (or empty, for a pure visual direction).

**Pairing (the important part):** this is **not** a passive per-sample sign flip. For a positive sample, the step looks up its negative twin by string-substituting the folder token in the image path and matching on identical latent shape — mirroring the dxqb logic:

```python
negative_path = batch['image_path'][0].replace(POSITIVE, NEGATIVE)
# find the batch/dataset item whose image_path == negative_path and whose latent shape matches
```

Training the **same content** in both directions (alpha `+1` on positive, `−1` on negative) makes the content-specific gradients cancel, so the LoRA learns only the *direction* between the two folders. A loose "flip sign on whatever image appears" would be far noisier and is explicitly rejected.

**Step logic:**
1. Determine sign from the path token → `set_multiplier(+1)` (positive) or `set_multiplier(-1)` (negative).
2. Run the **normal** Krea2 `predict` + `calculate_loss` on that sample.
3. Return the loss.

Runs at **batch_size = 1** (the pairing/sign logic assumes a homogeneous, single-item batch, matching dxqb's `batch['image_path'][0]`).

**Files touched:** `modules/module/LoRAModule.py` (`set_multiplier`), `modules/trainer/slider.py` (step), `modules/trainer/GenericTrainer.py` (gate), `modules/trainer/slider_config.py`.

## Component: Text slider (dataset mode)

**Concept:** a frozen teacher (LoRA off) synthesizes a guided target from prompt directions; the student (LoRA on) is trained to reproduce it. The dataset supplies the starting latents + timestep; the *direction* comes purely from the hardcoded prompts.

**One-time setup:** encode `TEXT_POSITIVE`, `TEXT_NEUTRAL`, `TEXT_NEGATIVE`, `TEXT_TARGET` once via `model.encode_text` and **cache** the resulting encoder hidden states (+ attention masks). Qwen3VL encoding is expensive; the slider prompts are fixed, so this is encoded once per run, not per step.

**Refactor required (the main porting effort):** factor a "forward the transformer given *supplied* text-encoder output, latent input, and timestep" helper out of `BaseKrea2Setup.predict` (`BaseKrea2Setup.py:69`). Today `predict` encodes the batch's own captions and computes everything inline; the slider needs to call the transformer-forward core with swapped-in cached prompt embeddings while reusing the same latent packing, timestep, and unpacking. Extract that core (roughly `BaseKrea2Setup.py:116`–`139`: pack → `model.transformer(...)` → unpack) into a reusable method; `predict` keeps calling it, and the slider calls it too.

**Step logic (dataset mode):**
1. Take the batch's latents; sample noise + a timestep exactly as `predict` normally does (reuse that portion).
2. **Teacher** (`set_multiplier(0)`), no grad, 3 forwards with cached embeds → `p` (positive), `n0` (neutral), `neg` (negative).
3. Compose target: `target = n0 + TEXT_GUIDANCE * (p - neg)`, detached.
4. **Student** (`set_multiplier(1)`), with grad, 1 forward conditioned on `TEXT_TARGET` embeds → `s`.
5. Loss = `MSE(s, target)`. Return it.

**Files touched:** `modules/modelSetup/BaseKrea2Setup.py` (extract forward core), `modules/model/Krea2Model.py` (optional prompt-embed cache field, if not kept in the slider module), `modules/trainer/slider.py` (step), `modules/trainer/GenericTrainer.py` (gate), `modules/trainer/slider_config.py`.

## Error handling

- **Image slider:** if a negative twin can't be found for a positive sample (missing file / shape mismatch), raise a clear error naming the offending path rather than silently skipping — a broken pair means a corrupt direction signal, which the user needs to know about.
- **Config validation:** `slider_config.py` values validated once at train start when `SLIDER_MODE` is set — non-`None` mode requires the corresponding fields to be non-empty; `TEXT_GUIDANCE` must be finite. Fail fast with a readable message.
- **Multiplier bounds:** `set_multiplier` accepts any float (negative and zero are intended). No clamping.

## Testing

Repo convention: **unittest, no pytest** (per project memory). Tests live under `tests/`.

Unit tests (fast, no model weights):
1. **`set_multiplier` scaling** — construct a small `LoRAModuleWrapper` over a toy `nn.Linear`, assert the forward delta scales linearly with the multiplier, and that `multiplier = 0` yields the base module's output, `-1` negates the delta.
2. **Text target composition** — pure-tensor test of `n0 + guidance*(p - neg)` given known inputs.
3. **Image sign selection** — given synthetic `image_path` strings, assert positive/negative classification and that `negative_path` substitution is correct.

Manual smoke (user-run, needs weights/hardware):
- Short Krea2 LoRA run with `SLIDER_MODE="image"` on a tiny paired dataset → loss moves, LoRA saves.
- Short Krea2 LoRA run with `SLIDER_MODE="text"` → loss moves, LoRA saves.
- Confirm `SLIDER_MODE=None` leaves the normal training code path unchanged (guard is a no-op).

## Open risks

- **`predict` extraction friction.** The whole text slider hinges on how cleanly the transformer-forward core factors out of `BaseKrea2Setup.predict`. If `predict` is more entangled than it looks (offload conductors, autocast contexts, attention masks tied to caption length), the extraction is the bulk of the work and may need adaptation.
- **Attention mask / seq-len assumptions.** Cached slider prompts have their own sequence lengths; the extracted forward must handle each prompt's mask correctly rather than assuming the batch caption's length.
- **Multiplier vs. non-LoRA modules.** `set_multiplier` is defined for plain LoRA only. Using a slider with DoRA/LoKr is unsupported and out of scope.
