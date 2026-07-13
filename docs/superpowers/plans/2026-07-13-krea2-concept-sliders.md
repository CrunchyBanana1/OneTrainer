# Krea2 Concept Sliders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hardcoded image and text concept sliders for Krea2 LoRA training, gated so normal training is unaffected.

**Architecture:** A shared `set_multiplier` on the LoRA wrapper drives both sliders (image: ±1 by folder; text: 0 teacher / 1 student). A hardcoded `slider_config.py` with a `SLIDER_MODE` switch, and a `slider.py` step module, are dispatched from a thin gate in `GenericTrainer`. The text slider reuses two helpers extracted from `BaseKrea2Setup.predict`.

**Tech Stack:** Python 3.11, PyTorch, diffusers (`Krea2Pipeline`, `Krea2Transformer2DModel`), mgds data pipeline, `unittest`.

## Global Constraints

- **Tests:** `unittest` only — **no pytest**. Files under `tests/`, classes `XxxTest(unittest.TestCase)`, end with `if __name__ == "__main__": unittest.main()`. Import production code as `from modules.… import …`.
- **Run tests** from repo root with the project venv: `venv/Scripts/python -m unittest tests.<pkg>.<module> -v` (Windows). `venv/bin/python` on Linux.
- **Lint:** global ruff config (`pyproject.toml`). Run `ruff check <file>` on changed files before each commit.
- **Branch:** `krea2-concept-sliders` (local; based on `pyside6-dataset-tool-lmstudio-pr`). No upstream PR.
- **Style:** match surrounding code. `multiplier` is a plain float attribute (never a registered buffer) so exported LoRAs are unaffected.
- **Scope:** Krea2 LoRA only. Plain LoRA modules only (DoRA/LoKr multiplier support is out of scope). Text slider is dataset-mode only. Image slider v1 is per-sample sign (no same-step twin pairing).
- **`tests/` needs package dirs:** add empty `tests/module/__init__.py` and `tests/trainer/__init__.py` if unittest discovery needs them (create alongside first test in each dir).

---

## File Structure

- **Modify** `modules/module/LoRAModule.py` — add `multiplier` to `PeftBase`, apply it in `LoRAModule.delta_forward`, add `LoRAModuleWrapper.set_multiplier`.
- **Create** `modules/trainer/slider_config.py` — hardcoded constants + `SLIDER_MODE` + `validate_slider_config()`.
- **Create** `modules/trainer/slider.py` — pure helpers (`image_multiplier_for_path`, `compose_text_target`), prompt cache, `image_slider_step`, `text_slider_step`, `slider_train_step` dispatcher.
- **Modify** `modules/modelSetup/BaseKrea2Setup.py` — extract `_prepare_noised_latent` and `_transformer_forward`; rebuild `predict` on them.
- **Modify** `modules/trainer/GenericTrainer.py` — gate the train step on `SLIDER_MODE`.
- **Create** tests under `tests/module/` and `tests/trainer/`.

---

### Task 1: LoRA multiplier primitive

**Files:**
- Modify: `modules/module/LoRAModule.py` (`PeftBase.__init__` ~L33-39; `LoRAModule.delta_forward` ~L578-581; `LoRAModuleWrapper.set_dropout` ~L1123 as template)
- Test: `tests/module/test_lora_multiplier.py`

**Interfaces:**
- Produces: `PeftBase.multiplier: float` (default `1.0`); `LoRAModule.delta_forward` scales its output by `self.multiplier`; `LoRAModuleWrapper.set_multiplier(multiplier: float) -> None` sets `module.multiplier` on every child module.

- [ ] **Step 1: Write the failing test**

```python
# tests/module/test_lora_multiplier.py
import unittest

import torch
from torch import nn

from modules.module.LoRAModule import LoRAModule, LoRAModuleWrapper


def _make_lora_over_linear(rank=4, alpha=4.0, in_f=8, out_f=8):
    orig = nn.Linear(in_f, out_f, bias=False)
    module = LoRAModule("test", orig, rank, alpha)
    # force a non-zero up projection so the delta is observable (lora_up inits to zero)
    nn.init.normal_(module.lora_up.weight, std=0.1)
    return orig, module


class LoraMultiplierTest(unittest.TestCase):
    def test_default_multiplier_is_one(self):
        _, module = _make_lora_over_linear()
        self.assertEqual(module.multiplier, 1.0)

    def test_delta_scales_linearly_with_multiplier(self):
        _, module = _make_lora_over_linear()
        x = torch.randn(2, 8)
        base_delta = module.delta_forward(x)
        module.multiplier = 2.0
        self.assertTrue(torch.allclose(module.delta_forward(x), base_delta * 2.0, atol=1e-6))

    def test_zero_multiplier_zeroes_delta(self):
        _, module = _make_lora_over_linear()
        x = torch.randn(2, 8)
        module.multiplier = 0.0
        self.assertTrue(torch.allclose(module.delta_forward(x), torch.zeros_like(module.delta_forward(x))))

    def test_negative_multiplier_negates_delta(self):
        _, module = _make_lora_over_linear()
        x = torch.randn(2, 8)
        base_delta = module.delta_forward(x)
        module.multiplier = -1.0
        self.assertTrue(torch.allclose(module.delta_forward(x), -base_delta, atol=1e-6))

    def test_wrapper_set_multiplier_propagates(self):
        orig = nn.Linear(8, 8, bias=False)
        model = nn.Module()
        model.layer = orig
        wrapper = LoRAModuleWrapper(model, "wrap", _DummyConfig())
        wrapper.set_multiplier(-1.0)
        for m in wrapper.lora_modules.values():
            self.assertEqual(m.multiplier, -1.0)


class _DummyConfig:
    # minimal stand-in for the LoRA config LoRAModuleWrapper reads
    lora_rank = 4
    lora_alpha = 4.0
    # NOTE: if LoRAModuleWrapper.__init__ needs more attributes, extend this
    # to match modules/util/config attributes read in its __init__.


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python -m unittest tests.module.test_lora_multiplier -v`
Expected: FAIL — `AttributeError: 'LoRAModule' object has no attribute 'multiplier'` (and `set_multiplier` missing).

> If `LoRAModuleWrapper(...)` construction in `test_wrapper_set_multiplier_propagates` fails due to config attributes, read `LoRAModuleWrapper.__init__` (`LoRAModule.py:829+`) and extend `_DummyConfig` with the exact attributes it reads. Keep the other four tests (which don't build a wrapper) as the primary gate.

- [ ] **Step 3: Add `multiplier` to `PeftBase.__init__`**

In `modules/module/LoRAModule.py`, in `PeftBase.__init__` (after `self._initialized = False`, ~L39):

```python
        self._initialized = False
        self.multiplier = 1.0  # slider control; plain float, never saved. Scales delta_forward output.
```

Also add to the class attribute annotations near the top of `PeftBase` (~L31):

```python
    _initialized: bool  # Tracks whether we've created the layers or not.
    multiplier: float   # slider multiplier applied to the LoRA delta (1.0 = normal)
```

- [ ] **Step 4: Apply the multiplier in `LoRAModule.delta_forward`**

Replace `LoRAModule.delta_forward` (~L578-581):

```python
    def delta_forward(self, x, *args, **kwargs) -> Tensor | None:
        self.check_initialized()
        ld = self.lora_up(self.dropout(self.lora_down(x)))
        return ld * (self.alpha / self.rank) * self.multiplier
```

- [ ] **Step 5: Add `set_multiplier` to `LoRAModuleWrapper`**

In `modules/module/LoRAModule.py`, right after `LoRAModuleWrapper.set_dropout` (~L1123-1130):

```python
    def set_multiplier(self, multiplier: float):
        """
        Sets the slider multiplier on every LoRA module (1.0 = normal, 0.0 = disabled,
        -1.0 = negated). Only honored by plain LoRAModule.delta_forward.
        """
        for module in self.lora_modules.values():
            module.multiplier = multiplier
```

- [ ] **Step 6: Run the tests and verify they pass**

Run: `venv/Scripts/python -m unittest tests.module.test_lora_multiplier -v`
Expected: PASS (all tests).

- [ ] **Step 7: Lint and commit**

```bash
ruff check modules/module/LoRAModule.py tests/module/test_lora_multiplier.py
git add modules/module/LoRAModule.py tests/module/test_lora_multiplier.py tests/module/__init__.py
git commit -m "feat(lora): add multiplier control for concept sliders"
```

---

### Task 2: Slider config module

**Files:**
- Create: `modules/trainer/slider_config.py`
- Test: `tests/trainer/test_slider_config.py`

**Interfaces:**
- Produces:
  - Constants: `SLIDER_MODE` (`None | "image" | "text"`), `IMAGE_POSITIVE_TOKEN: str`, `IMAGE_NEGATIVE_TOKEN: str`, `TEXT_POSITIVE: str`, `TEXT_NEUTRAL: str`, `TEXT_NEGATIVE: str`, `TEXT_TARGET: str`, `TEXT_GUIDANCE: float`.
  - `validate_slider_config() -> None` — raises `ValueError` with a readable message when `SLIDER_MODE` is set but required fields are empty/invalid. No-op when `SLIDER_MODE is None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/trainer/test_slider_config.py
import unittest
from unittest import mock

from modules.trainer import slider_config


class SliderConfigValidationTest(unittest.TestCase):
    def test_none_mode_is_noop(self):
        with mock.patch.object(slider_config, "SLIDER_MODE", None):
            slider_config.validate_slider_config()  # must not raise

    def test_image_mode_requires_tokens(self):
        with mock.patch.object(slider_config, "SLIDER_MODE", "image"), \
             mock.patch.object(slider_config, "IMAGE_POSITIVE_TOKEN", ""):
            with self.assertRaises(ValueError):
                slider_config.validate_slider_config()

    def test_text_mode_requires_prompts(self):
        with mock.patch.object(slider_config, "SLIDER_MODE", "text"), \
             mock.patch.object(slider_config, "TEXT_POSITIVE", ""):
            with self.assertRaises(ValueError):
                slider_config.validate_slider_config()

    def test_text_mode_requires_finite_guidance(self):
        with mock.patch.object(slider_config, "SLIDER_MODE", "text"), \
             mock.patch.object(slider_config, "TEXT_GUIDANCE", float("inf")):
            with self.assertRaises(ValueError):
                slider_config.validate_slider_config()

    def test_unknown_mode_raises(self):
        with mock.patch.object(slider_config, "SLIDER_MODE", "bogus"):
            with self.assertRaises(ValueError):
                slider_config.validate_slider_config()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python -m unittest tests.trainer.test_slider_config -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modules.trainer.slider_config'`.

- [ ] **Step 3: Create the config module**

```python
# modules/trainer/slider_config.py
"""
Hardcoded configuration for Krea2 concept sliders.

Edit the constants below and set SLIDER_MODE to enable slider training.
When SLIDER_MODE is None, normal training is completely unaffected.

This is intentionally NOT wired into the UI — it is a local training tool.
"""
import math

# None | "image" | "text"
SLIDER_MODE: str | None = None

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
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `venv/Scripts/python -m unittest tests.trainer.test_slider_config -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
ruff check modules/trainer/slider_config.py tests/trainer/test_slider_config.py
git add modules/trainer/slider_config.py tests/trainer/test_slider_config.py tests/trainer/__init__.py
git commit -m "feat(slider): add hardcoded slider config with validation"
```

---

### Task 3: Slider pure helpers (sign + target math)

**Files:**
- Create: `modules/trainer/slider.py` (helpers only in this task)
- Test: `tests/trainer/test_slider_helpers.py`

**Interfaces:**
- Produces:
  - `image_multiplier_for_path(image_path: str, positive_token: str, negative_token: str) -> float` — returns `+1.0` if `positive_token` in path, `-1.0` if `negative_token` in path; raises `ValueError` if neither (or both) present.
  - `compose_text_target(neutral: Tensor, positive: Tensor, negative: Tensor, guidance: float) -> Tensor` — returns `neutral + guidance * (positive - negative)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/trainer/test_slider_helpers.py
import unittest

import torch

from modules.trainer.slider import compose_text_target, image_multiplier_for_path


class ImageMultiplierTest(unittest.TestCase):
    def test_positive_path_is_plus_one(self):
        self.assertEqual(image_multiplier_for_path("/data/positive/foo.png", "positive", "negative"), 1.0)

    def test_negative_path_is_minus_one(self):
        self.assertEqual(image_multiplier_for_path("/data/negative/foo.png", "positive", "negative"), -1.0)

    def test_neither_token_raises(self):
        with self.assertRaises(ValueError):
            image_multiplier_for_path("/data/other/foo.png", "positive", "negative")

    def test_both_tokens_raises(self):
        with self.assertRaises(ValueError):
            image_multiplier_for_path("/data/positive/negative_foo.png", "positive", "negative")


class ComposeTextTargetTest(unittest.TestCase):
    def test_composition_formula(self):
        neutral = torch.zeros(2, 2)
        positive = torch.ones(2, 2)
        negative = torch.full((2, 2), 0.25)
        out = compose_text_target(neutral, positive, negative, guidance=2.0)
        # 0 + 2 * (1 - 0.25) = 1.5
        self.assertTrue(torch.allclose(out, torch.full((2, 2), 1.5)))

    def test_zero_guidance_returns_neutral(self):
        neutral = torch.randn(3, 3)
        out = compose_text_target(neutral, torch.randn(3, 3), torch.randn(3, 3), guidance=0.0)
        self.assertTrue(torch.allclose(out, neutral))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python -m unittest tests.trainer.test_slider_helpers -v`
Expected: FAIL — `ImportError: cannot import name 'compose_text_target' from 'modules.trainer.slider'`.

- [ ] **Step 3: Create `slider.py` with the helpers**

```python
# modules/trainer/slider.py
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
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `venv/Scripts/python -m unittest tests.trainer.test_slider_helpers -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
ruff check modules/trainer/slider.py tests/trainer/test_slider_helpers.py
git add modules/trainer/slider.py tests/trainer/test_slider_helpers.py
git commit -m "feat(slider): add image sign + text target helpers"
```

---

### Task 4: Extract forward helpers from Krea2 predict

**Files:**
- Modify: `modules/modelSetup/BaseKrea2Setup.py` (`predict` L69-163)
- Test: `tests/module/test_krea2_transformer_forward.py`

**Interfaces:**
- Produces (methods on `BaseKrea2Setup`):
  - `_prepare_noised_latent(self, model, batch, config, generator, deterministic) -> dict` with keys `scaled_latent_image`, `latent_noise`, `timestep`, `scaled_noisy_latent_image`, `sigma`.
  - `_transformer_forward(self, model, latent_input, text_encoder_output, text_attention_mask, timestep) -> Tensor` (unpacked predicted flow). Recomputes `position_ids` and collapses an all-true mask to `None`, per call.
- Consumes: nothing new; `predict` is rebuilt to call both.

**Behavior must be preserved:** `predict` returns the same dict as before.

- [ ] **Step 1: Write the failing wiring test (stubbed model)**

```python
# tests/module/test_krea2_transformer_forward.py
import unittest

import torch

from modules.modelSetup.BaseKrea2Setup import BaseKrea2Setup


class _DType:
    def torch_dtype(self):
        return torch.float32


class _StubModel:
    """Minimal stand-in exercising _transformer_forward's wiring."""
    def __init__(self):
        self.train_dtype = _DType()
        self.captured = {}

    def pack_latents(self, x):
        return x  # identity for the test

    def unpack_latents(self, x, height, width):
        self.captured["unpack_hw"] = (height, width)
        return x  # identity

    def transformer(self, hidden_states, encoder_hidden_states, timestep,
                    position_ids, encoder_attention_mask, return_dict):
        self.captured["encoder_attention_mask"] = encoder_attention_mask
        self.captured["timestep"] = timestep
        return (hidden_states,)  # echo


class Krea2TransformerForwardTest(unittest.TestCase):
    def _setup(self):
        # construct without running __init__ (avoids heavy deps); set train_device only
        setup = BaseKrea2Setup.__new__(BaseKrea2Setup)
        setup.train_device = torch.device("cpu")
        return setup

    def test_all_true_mask_collapses_to_none_and_echoes_input(self):
        setup = self._setup()
        model = _StubModel()
        latent = torch.randn(1, 4, 8, 8)                 # (B,C,H,W), H,W even
        text_out = torch.randn(1, 5, 16)                 # (B,T,E)
        mask = torch.ones(1, 5, dtype=torch.bool)        # all-true -> None
        out = setup._transformer_forward(model, latent, text_out, mask, timestep=torch.tensor([500.0]))
        self.assertIsNone(model.captured["encoder_attention_mask"])
        self.assertEqual(out.shape, latent.shape)
        self.assertTrue(torch.allclose(out, latent))

    def test_timestep_is_scaled_by_1000(self):
        setup = self._setup()
        model = _StubModel()
        latent = torch.randn(1, 4, 8, 8)
        text_out = torch.randn(1, 5, 16)
        mask = torch.ones(1, 5, dtype=torch.bool)
        setup._transformer_forward(model, latent, text_out, mask, timestep=torch.tensor([1000.0]))
        self.assertTrue(torch.allclose(model.captured["timestep"], torch.tensor([1.0])))


if __name__ == "__main__":
    unittest.main()
```

> If `Krea2Pipeline.prepare_position_ids` cannot run on the stub shapes, patch it in the test with `unittest.mock.patch("modules.modelSetup.BaseKrea2Setup.Krea2Pipeline.prepare_position_ids", return_value=None)`. Confirm the real import name by reading the top of `BaseKrea2Setup.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python -m unittest tests.module.test_krea2_transformer_forward -v`
Expected: FAIL — `AttributeError: 'BaseKrea2Setup' object has no attribute '_transformer_forward'`.

- [ ] **Step 3: Add `_transformer_forward` (moved from predict L116-142)**

Add this method to `BaseKrea2Setup` (near `predict`). It contains the exact logic currently inline in `predict`, made mask-None-safe:

```python
    def _transformer_forward(
            self,
            model: Krea2Model,
            latent_input: Tensor,
            text_encoder_output: Tensor,
            text_attention_mask: Tensor | None,
            timestep: Tensor,
    ) -> Tensor:
        packed_latent_input = model.pack_latents(latent_input)

        # position ids: text tokens at origin, image tokens at latent-grid coords (patch_size = 2)
        text_seq_len = text_encoder_output.shape[1]
        grid_height = latent_input.shape[-2] // 2
        grid_width = latent_input.shape[-1] // 2
        position_ids = Krea2Pipeline.prepare_position_ids(
            text_seq_len, grid_height, grid_width, self.train_device
        )

        if text_attention_mask is not None and torch.all(text_attention_mask):
            text_attention_mask = None

        packed_predicted_flow = model.transformer(
            hidden_states=packed_latent_input.to(dtype=model.train_dtype.torch_dtype()),
            encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
            timestep=timestep / 1000,
            position_ids=position_ids,
            encoder_attention_mask=text_attention_mask,
            return_dict=False,
        )[0]

        return model.unpack_latents(
            packed_predicted_flow,
            height=latent_input.shape[-2],
            width=latent_input.shape[-1],
        )
```

- [ ] **Step 4: Add `_prepare_noised_latent` (moved from predict L94-113)**

```python
    def _prepare_noised_latent(
            self,
            model: Krea2Model,
            batch: dict,
            config: TrainConfig,
            generator: torch.Generator,
            deterministic: bool,
    ) -> dict:
        latent_image = batch['latent_image']
        scaled_latent_image = model.scale_latents(latent_image)
        latent_noise = self._create_noise(scaled_latent_image, config, generator)

        shift = model.calculate_timestep_shift(scaled_latent_image.shape[-2], scaled_latent_image.shape[-1])
        timestep = self._get_timestep_discrete(
            model.noise_scheduler.config['num_train_timesteps'],
            deterministic,
            generator,
            scaled_latent_image.shape[0],
            config,
            shift=shift if config.dynamic_timestep_shifting else config.timestep_shift,
        )

        scaled_noisy_latent_image, sigma = self._add_noise_discrete(
            scaled_latent_image,
            latent_noise,
            timestep,
            model.noise_scheduler.timesteps,
        )

        return {
            'scaled_latent_image': scaled_latent_image,
            'latent_noise': latent_noise,
            'timestep': timestep,
            'scaled_noisy_latent_image': scaled_noisy_latent_image,
            'sigma': sigma,
        }
```

- [ ] **Step 5: Rebuild `predict` on the two helpers**

Replace the body of `predict` (keep the signature and the outer `with model.autocast_context:` and seeding) so it reads:

```python
        with model.autocast_context:
            batch_seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)
            rand = Random(batch_seed)

            text_encoder_output, text_attention_mask = model.encode_text(
                train_device=self.train_device,
                batch_size=batch['latent_image'].shape[0],
                rand=rand,
                tokens=batch.get("tokens"),
                tokens_mask=batch.get("tokens_mask"),
                text_encoder_output=batch.get('text_encoder_hidden_state'),
                text_encoder_dropout_probability=config.text_encoder.dropout_probability if not deterministic else None,
            )

            prep = self._prepare_noised_latent(model, batch, config, generator, deterministic)

            predicted_flow = self._transformer_forward(
                model,
                prep['scaled_noisy_latent_image'],
                text_encoder_output,
                text_attention_mask,
                prep['timestep'],
            )

            flow = prep['latent_noise'] - prep['scaled_latent_image']
            model_output_data = {
                'loss_type': 'target',
                'timestep': prep['timestep'],
                'predicted': predicted_flow,
                'target': flow,
            }

            if config.debug_mode:
                with torch.no_grad():
                    predicted_scaled_latent_image = prep['scaled_noisy_latent_image'] - predicted_flow * prep['sigma']
                    self._save_tokens("7-prompt", batch['tokens'], model.tokenizer, config, train_progress)
                    self._save_latent("1-noise", prep['latent_noise'], config, train_progress)
                    self._save_latent("2-noisy_image", prep['scaled_noisy_latent_image'], config, train_progress)
                    self._save_latent("3-predicted_flow", predicted_flow, config, train_progress)
                    self._save_latent("4-flow", flow, config, train_progress)
                    self._save_latent("5-predicted_image", predicted_scaled_latent_image, config, train_progress)
                    self._save_latent("6-image", prep['scaled_latent_image'], config, train_progress)

        return model_output_data
```

- [ ] **Step 6: Run the tests and verify they pass**

Run: `venv/Scripts/python -m unittest tests.module.test_krea2_transformer_forward -v`
Expected: PASS.

- [ ] **Step 7: Sanity-check the refactor didn't break imports**

Run: `venv/Scripts/python -c "import modules.modelSetup.Krea2LoRASetup"`
Expected: no error.

- [ ] **Step 8: Lint and commit**

```bash
ruff check modules/modelSetup/BaseKrea2Setup.py tests/module/test_krea2_transformer_forward.py
git add modules/modelSetup/BaseKrea2Setup.py tests/module/test_krea2_transformer_forward.py
git commit -m "refactor(krea2): extract prepare/forward helpers from predict"
```

---

### Task 5: Slider steps + dispatcher

**Files:**
- Modify: `modules/trainer/slider.py` (add prompt cache, `image_slider_step`, `text_slider_step`, `slider_train_step`)
- Test: `tests/trainer/test_slider_steps.py`

**Interfaces:**
- Consumes: `image_multiplier_for_path`, `compose_text_target` (Task 3); `model_setup._prepare_noised_latent`, `model_setup._transformer_forward` (Task 4); `model.transformer_lora.set_multiplier` (Task 1); `slider_config` (Task 2); `model.encode_text(train_device, text=...) -> (Tensor, Tensor)`.
- Produces:
  - `SliderPromptCache` with attributes `positive`, `neutral`, `negative`, `target`, each a `(output, mask)` tuple; `get_prompt_cache(model, train_device) -> SliderPromptCache` (memoized module-level).
  - `image_slider_step(model_setup, model, batch, config, train_progress) -> Tensor`
  - `text_slider_step(model_setup, model, batch, config, train_progress) -> Tensor`
  - `slider_train_step(model_setup, model, batch, config, train_progress) -> Tensor` — dispatches on `slider_config.SLIDER_MODE`.

- [ ] **Step 1: Write the failing test (stubbed setup/model)**

```python
# tests/trainer/test_slider_steps.py
import unittest
from unittest import mock

import torch

from modules.trainer import slider, slider_config


class _StubLora:
    def __init__(self):
        self.multipliers = []
    def set_multiplier(self, m):
        self.multipliers.append(m)


class _StubModel:
    def __init__(self):
        self.transformer_lora = _StubLora()
        self.autocast_context = _nullcontext()


class _nullcontext:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _StubSetup:
    """Returns a fixed noised latent and a prompt-keyed forward result."""
    def __init__(self, forward_map):
        self.train_device = torch.device("cpu")
        self._forward_map = forward_map
        self.predict_calls = 0
    def _prepare_noised_latent(self, model, batch, config, generator, deterministic):
        return {'scaled_noisy_latent_image': torch.zeros(1, 4, 8, 8),
                'timestep': torch.tensor([500.0]),
                'latent_noise': torch.zeros(1, 4, 8, 8),
                'scaled_latent_image': torch.zeros(1, 4, 8, 8),
                'sigma': torch.tensor([0.5])}
    def _transformer_forward(self, model, latent_input, text_out, mask, timestep):
        # identify prompt by the sentinel value we stored in text_out[0,0,0]
        key = int(text_out[0, 0, 0].item())
        return self._forward_map[key]
    def predict(self, model, batch, config, train_progress):
        self.predict_calls += 1
        return {'predicted': torch.ones(1, 4, 8, 8), 'target': torch.zeros(1, 4, 8, 8)}
    def calculate_loss(self, model, batch, data, config):
        return torch.tensor(0.7)


def _sentinel_prompt(value):
    t = torch.zeros(1, 3, 4)
    t[0, 0, 0] = value
    return (t, torch.ones(1, 3, dtype=torch.bool))


class TextSliderStepTest(unittest.TestCase):
    def test_teacher_zero_student_one_and_mse_target(self):
        # prompt sentinels: pos=1, neu=2, neg=3, target=4
        cache = slider.SliderPromptCache(
            positive=_sentinel_prompt(1), neutral=_sentinel_prompt(2),
            negative=_sentinel_prompt(3), target=_sentinel_prompt(4),
        )
        forward_map = {
            1: torch.full((1, 4, 8, 8), 1.0),   # positive teacher pred
            2: torch.full((1, 4, 8, 8), 0.0),   # neutral teacher pred
            3: torch.full((1, 4, 8, 8), 0.25),  # negative teacher pred
            4: torch.full((1, 4, 8, 8), 0.5),   # student pred (target prompt)
        }
        setup = _StubSetup(forward_map)
        model = _StubModel()
        with mock.patch.object(slider, "get_prompt_cache", return_value=cache), \
             mock.patch.object(slider_config, "TEXT_GUIDANCE", 2.0):
            loss = slider.text_slider_step(setup, model, batch={}, config=_Cfg(), train_progress=_TP())
        # target = 0 + 2*(1 - 0.25) = 1.5 ; student = 0.5 ; mse = (1.0)^2 = 1.0
        self.assertAlmostEqual(loss.item(), 1.0, places=5)
        # teacher passes set multiplier 0 (x3), student sets 1 (x1), in that order
        self.assertEqual(model.transformer_lora.multipliers, [0.0, 0.0, 0.0, 1.0])


class ImageSliderStepTest(unittest.TestCase):
    def test_positive_sets_plus_one(self):
        setup = _StubSetup({})
        model = _StubModel()
        with mock.patch.object(slider_config, "IMAGE_POSITIVE_TOKEN", "positive"), \
             mock.patch.object(slider_config, "IMAGE_NEGATIVE_TOKEN", "negative"):
            loss = slider.image_slider_step(setup, model, batch={'image_path': ['/d/positive/a.png']},
                                            config=_Cfg(), train_progress=_TP())
        self.assertEqual(model.transformer_lora.multipliers, [1.0])
        self.assertEqual(setup.predict_calls, 1)
        self.assertAlmostEqual(loss.item(), 0.7, places=5)

    def test_negative_sets_minus_one(self):
        setup = _StubSetup({})
        model = _StubModel()
        with mock.patch.object(slider_config, "IMAGE_POSITIVE_TOKEN", "positive"), \
             mock.patch.object(slider_config, "IMAGE_NEGATIVE_TOKEN", "negative"):
            slider.image_slider_step(setup, model, batch={'image_path': ['/d/negative/a.png']},
                                     config=_Cfg(), train_progress=_TP())
        self.assertEqual(model.transformer_lora.multipliers, [-1.0])


class DispatchTest(unittest.TestCase):
    def test_dispatch_image(self):
        with mock.patch.object(slider_config, "SLIDER_MODE", "image"), \
             mock.patch.object(slider, "image_slider_step", return_value=torch.tensor(1.0)) as img, \
             mock.patch.object(slider, "text_slider_step", return_value=torch.tensor(2.0)) as txt:
            out = slider.slider_train_step("setup", "model", {}, "cfg", "tp")
        img.assert_called_once()
        txt.assert_not_called()
        self.assertEqual(out.item(), 1.0)

    def test_dispatch_text(self):
        with mock.patch.object(slider_config, "SLIDER_MODE", "text"), \
             mock.patch.object(slider, "image_slider_step", return_value=torch.tensor(1.0)) as img, \
             mock.patch.object(slider, "text_slider_step", return_value=torch.tensor(2.0)) as txt:
            out = slider.slider_train_step("setup", "model", {}, "cfg", "tp")
        txt.assert_called_once()
        img.assert_not_called()
        self.assertEqual(out.item(), 2.0)


class _Cfg:
    train_device = torch.device("cpu")

class _TP:
    global_step = 0


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python -m unittest tests.trainer.test_slider_steps -v`
Expected: FAIL — `AttributeError: module 'modules.trainer.slider' has no attribute 'SliderPromptCache'`.

- [ ] **Step 3: Add the prompt cache + steps + dispatcher to `slider.py`**

Append to `modules/trainer/slider.py` (keep the Task-3 helpers above):

```python
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from modules.trainer import slider_config

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

        model.transformer_lora.set_multiplier(0.0)  # teacher: LoRA off
        with torch.no_grad():
            pos = model_setup._transformer_forward(model, latent_input, cache.positive[0], cache.positive[1], timestep)
            neu = model_setup._transformer_forward(model, latent_input, cache.neutral[0], cache.neutral[1], timestep)
            neg = model_setup._transformer_forward(model, latent_input, cache.negative[0], cache.negative[1], timestep)
            target = compose_text_target(neu, pos, neg, slider_config.TEXT_GUIDANCE).detach()

        model.transformer_lora.set_multiplier(1.0)  # student: LoRA on
        student = model_setup._transformer_forward(model, latent_input, cache.target[0], cache.target[1], timestep)
        loss = F.mse_loss(student.float(), target.float())
    return loss


def slider_train_step(model_setup, model, batch, config, train_progress):
    if slider_config.SLIDER_MODE == "image":
        return image_slider_step(model_setup, model, batch, config, train_progress)
    if slider_config.SLIDER_MODE == "text":
        return text_slider_step(model_setup, model, batch, config, train_progress)
    raise ValueError(f"slider_train_step called with SLIDER_MODE={slider_config.SLIDER_MODE!r}")
```

> Note: `image_multiplier_for_path` and `compose_text_target` are already defined at the top of this file (Task 3). Do not re-import them.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `venv/Scripts/python -m unittest tests.trainer.test_slider_steps -v`
Expected: PASS (all classes).

- [ ] **Step 5: Lint and commit**

```bash
ruff check modules/trainer/slider.py tests/trainer/test_slider_steps.py
git add modules/trainer/slider.py tests/trainer/test_slider_steps.py
git commit -m "feat(slider): add image/text slider steps and dispatcher"
```

---

### Task 6: Gate GenericTrainer on SLIDER_MODE

**Files:**
- Modify: `modules/trainer/GenericTrainer.py` (train step ~L736-753; add `validate_slider_config()` call at train start)
- Test: `tests/trainer/test_slider_gate.py`

**Interfaces:**
- Consumes: `slider.slider_train_step`, `slider_config.SLIDER_MODE`, `slider_config.validate_slider_config`.
- Produces: no new public API. When `SLIDER_MODE is None`, the existing train-step block runs unchanged.

- [ ] **Step 1: Write the failing test**

This test verifies the gate function used by the trainer picks the slider path only when a mode is set, without constructing a full trainer. Factor the decision into a tiny module-level helper so it is unit-testable.

```python
# tests/trainer/test_slider_gate.py
import unittest
from unittest import mock

from modules.trainer import slider_config
from modules.trainer.slider import slider_enabled


class SliderEnabledTest(unittest.TestCase):
    def test_disabled_when_none(self):
        with mock.patch.object(slider_config, "SLIDER_MODE", None):
            self.assertFalse(slider_enabled())

    def test_enabled_for_image(self):
        with mock.patch.object(slider_config, "SLIDER_MODE", "image"):
            self.assertTrue(slider_enabled())

    def test_enabled_for_text(self):
        with mock.patch.object(slider_config, "SLIDER_MODE", "text"):
            self.assertTrue(slider_enabled())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python -m unittest tests.trainer.test_slider_gate -v`
Expected: FAIL — `ImportError: cannot import name 'slider_enabled'`.

- [ ] **Step 3: Add `slider_enabled` to `slider.py`**

```python
def slider_enabled() -> bool:
    return slider_config.SLIDER_MODE is not None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python -m unittest tests.trainer.test_slider_gate -v`
Expected: PASS.

- [ ] **Step 5: Add the import and gate in `GenericTrainer.py`**

Add near the other `from modules.trainer…` imports at the top of the file:

```python
from modules.trainer import slider
from modules.trainer.slider_config import validate_slider_config
```

In `GenericTrainer.train` (early, right after the method starts setting up — a safe spot is just before the epoch loop), add a one-time validation so a misconfigured slider fails fast:

```python
        validate_slider_config()
```

Then wrap the existing prediction block. Find (~L736-753):

```python
                    prior_pred_indices = [i for i in range(self.config.batch_size)
                                          if ConceptType(batch['concept_type'][i]) == ConceptType.PRIOR_PREDICTION]
                    if len(prior_pred_indices) > 0 \
                            ...
                    else:
                        model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)

                    loss = self.model_setup.calculate_loss(self.model, batch, model_output_data, self.config)
```

Replace it with a top-level branch that leaves the normal path byte-identical inside the `else`:

```python
                    if slider.slider_enabled():
                        loss = slider.slider_train_step(self.model_setup, self.model, batch, self.config, train_progress)
                    else:
                        prior_pred_indices = [i for i in range(self.config.batch_size)
                                              if ConceptType(batch['concept_type'][i]) == ConceptType.PRIOR_PREDICTION]
                        if len(prior_pred_indices) > 0 \
                                or (self.config.masked_training
                                    and self.config.masked_prior_preservation_weight > 0
                                    and self.config.training_method == TrainingMethod.LORA):
                            with self.model_setup.prior_model(self.model, self.config), torch.no_grad():
                                #do NOT create a subbatch using the indices, even though it would be more efficient:
                                #different timesteps are used for a smaller subbatch by predict(), but the conditioning must match exactly:
                                prior_model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)
                            model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)
                            prior_model_prediction = prior_model_output_data['predicted'].to(dtype=model_output_data['target'].dtype)
                            model_output_data['target'][prior_pred_indices] = prior_model_prediction[prior_pred_indices]
                            model_output_data['prior_target'] = prior_model_prediction
                        else:
                            model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)

                        loss = self.model_setup.calculate_loss(self.model, batch, model_output_data, self.config)
```

> Preserve the exact indentation of the surrounding `with` block. The only change is the new outer `if slider.slider_enabled(): … else:` wrapper.

- [ ] **Step 6: Verify normal training path is unchanged (import + full suite)**

Run: `venv/Scripts/python -c "import modules.trainer.GenericTrainer"`
Expected: no error.

Run the full slider suite:
`venv/Scripts/python -m unittest tests.module.test_lora_multiplier tests.module.test_krea2_transformer_forward tests.trainer.test_slider_config tests.trainer.test_slider_helpers tests.trainer.test_slider_steps tests.trainer.test_slider_gate -v`
Expected: PASS (all).

- [ ] **Step 7: Lint and commit**

```bash
ruff check modules/trainer/GenericTrainer.py modules/trainer/slider.py tests/trainer/test_slider_gate.py
git add modules/trainer/GenericTrainer.py modules/trainer/slider.py tests/trainer/test_slider_gate.py
git commit -m "feat(slider): gate GenericTrainer train step on SLIDER_MODE"
```

---

### Task 7: Docs + manual smoke checklist

**Files:**
- Create: `docs/Sliders.md` (usage notes; not wired to UI)

**Interfaces:** none (documentation).

- [ ] **Step 1: Write the usage doc**

```markdown
# Krea2 Concept Sliders (local tool)

Hardcoded image/text sliders for Krea2 LoRA training. Configure in
`modules/trainer/slider_config.py`, then run a normal Krea2 LoRA training.
Set `SLIDER_MODE = None` to disable (default).

## Layer targeting recommendation ("leave text alone")
For cleaner sliders, target the LoRA at visual layers (self-attention, conv,
time-embedding) and exclude text-conditioning layers, via your LoRA layer
selection in the training config.

## Image slider
- Set `SLIDER_MODE = "image"`. Run at **batch_size = 1**.
- Dataset: one concept with parallel subfolders whose paths contain
  `IMAGE_POSITIVE_TOKEN` / `IMAGE_NEGATIVE_TOKEN` (default `positive`/`negative`),
  holding same-named image pairs. Captions can match or be empty.
- v1 uses per-sample sign (no same-step twin pairing). Use gradient
  accumulation across pairs to reduce variance.

## Text slider (dataset mode)
- Set `SLIDER_MODE = "text"`. Provide any Krea2 dataset for starting latents.
- Set `TEXT_POSITIVE`, `TEXT_NEUTRAL`, `TEXT_NEGATIVE`, `TEXT_TARGET`,
  `TEXT_GUIDANCE`. `TEXT_TARGET` is usually identical to `TEXT_NEUTRAL`.
- Loss = MSE(student, neutral + guidance*(positive - negative)).

## Known limitations
- Plain LoRA only (DoRA/LoKr multiplier unsupported).
- Quantized transformer: the multiplier is not applied on the
  `forward_with_lora` path (`LoRAModule.py`); use a non-quantized run.
```

- [ ] **Step 2: Manual smoke test (user-run, needs weights/GPU)**

Not automated. Perform each and confirm:
1. `SLIDER_MODE=None`: start a normal Krea2 LoRA run for a few steps → trains as before (regression check).
2. `SLIDER_MODE="image"` at batch_size=1 on a tiny paired dataset → loss is finite and a LoRA saves.
3. `SLIDER_MODE="text"` with prompts set → loss is finite and a LoRA saves.

- [ ] **Step 3: Commit**

```bash
git add docs/Sliders.md
git commit -m "docs: add Krea2 slider usage notes"
```

---

## Self-Review

- **Spec coverage:**
  - Shared `set_multiplier` primitive → Task 1. ✓
  - Hardcoded config + `SLIDER_MODE` gate → Tasks 2, 6. ✓
  - Image slider (per-sample sign v1) → Tasks 3, 5. ✓
  - Text slider (dataset mode, teacher/student, prompt cache) → Tasks 3, 4, 5. ✓
  - `predict` forward-core extraction (R1) → Task 4. ✓
  - Per-prompt mask/seq-len (R2) → handled inside `_transformer_forward` (Task 4). ✓
  - "Leave text alone" recommendation → Task 7 doc. ✓
  - Quantized `forward_with_lora` residual → documented limitation, Task 7. ✓ (out of scope to implement)
- **Placeholder scan:** none — every code/test step has full content.
- **Type consistency:** `set_multiplier(float)`, `image_multiplier_for_path(str,str,str)->float`, `compose_text_target(Tensor,Tensor,Tensor,float)->Tensor`, `_transformer_forward(...)->Tensor`, `_prepare_noised_latent(...)->dict`, `SliderPromptCache(positive/neutral/negative/target: tuple)`, `slider_train_step(model_setup,model,batch,config,train_progress)->Tensor` — consistent across Tasks 1–6.
