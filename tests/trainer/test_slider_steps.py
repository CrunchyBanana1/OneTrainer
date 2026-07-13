# tests/trainer/test_slider_steps.py
import unittest
from unittest import mock

from modules.trainer import slider, slider_config

import torch


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
        self.seed_overrides = []
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
    def predict(self, model, batch, config, train_progress, *, seed_override=None):
        self.predict_calls += 1
        self.seed_overrides.append(seed_override)
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
        # multiplier is set to +1 for the step, then restored to the neutral 1.0 in `finally`.
        self.assertEqual(model.transformer_lora.multipliers, [1.0, 1.0])
        self.assertEqual(setup.predict_calls, 1)
        self.assertAlmostEqual(loss.item(), 0.7, places=5)

    def test_negative_sets_minus_one(self):
        setup = _StubSetup({})
        model = _StubModel()
        with mock.patch.object(slider_config, "IMAGE_POSITIVE_TOKEN", "positive"), \
             mock.patch.object(slider_config, "IMAGE_NEGATIVE_TOKEN", "negative"):
            slider.image_slider_step(setup, model, batch={'image_path': ['/d/negative/a.png']},
                                     config=_Cfg(), train_progress=_TP())
        # multiplier is set to -1 for the step, then restored to the neutral 1.0 in `finally`.
        self.assertEqual(model.transformer_lora.multipliers, [-1.0, 1.0])

    def test_multiplier_restored_to_one_after_step(self):
        setup = _StubSetup({})
        model = _StubModel()
        with mock.patch.object(slider_config, "IMAGE_POSITIVE_TOKEN", "positive"), \
             mock.patch.object(slider_config, "IMAGE_NEGATIVE_TOKEN", "negative"):
            slider.image_slider_step(setup, model, batch={'image_path': ['/d/negative/a.png']},
                                     config=_Cfg(), train_progress=_TP())
        self.assertEqual(model.transformer_lora.multipliers[-1], 1.0)

    def test_twins_share_pair_aligned_seed(self):
        # A positive on an even global_step and its negative twin on the next (odd) step must
        # receive the SAME seed_override (global_step // 2), so they get identical noise + timestep.
        setup = _StubSetup({})
        model = _StubModel()

        class _TPStep:
            def __init__(self, step):
                self.global_step = step

        with mock.patch.object(slider_config, "IMAGE_POSITIVE_TOKEN", "positive"), \
             mock.patch.object(slider_config, "IMAGE_NEGATIVE_TOKEN", "negative"):
            slider.image_slider_step(setup, model, batch={'image_path': ['/d/positive/a.png']},
                                     config=_Cfg(), train_progress=_TPStep(4))   # even step -> 4//2 = 2
            slider.image_slider_step(setup, model, batch={'image_path': ['/d/negative/a.png']},
                                     config=_Cfg(), train_progress=_TPStep(5))   # odd step  -> 5//2 = 2
        self.assertEqual(setup.seed_overrides, [2, 2])

    def test_raises_on_batch_size_greater_than_one(self):
        setup = _StubSetup({})
        model = _StubModel()
        with mock.patch.object(slider_config, "IMAGE_POSITIVE_TOKEN", "positive"), \
             mock.patch.object(slider_config, "IMAGE_NEGATIVE_TOKEN", "negative"), \
             self.assertRaises(ValueError):
            slider.image_slider_step(
                setup, model,
                batch={'image_path': ['/d/positive/a.png', '/d/negative/b.png']},
                config=_Cfg(), train_progress=_TP(),
            )
        # must fail before touching the multiplier or running predict/calculate_loss.
        self.assertEqual(model.transformer_lora.multipliers, [])
        self.assertEqual(setup.predict_calls, 0)


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

    def test_dispatch_default_raises(self):
        with mock.patch.object(slider_config, "SLIDER_MODE", None), self.assertRaises(ValueError):
            slider.slider_train_step("setup", "model", {}, "cfg", "tp")

    def test_dispatch_bogus_raises(self):
        with mock.patch.object(slider_config, "SLIDER_MODE", "bogus"), self.assertRaises(ValueError):
            slider.slider_train_step("setup", "model", {}, "cfg", "tp")


class _StubModelForCache:
    """Counts encode_text calls and returns a distinct object per call."""
    def __init__(self):
        self.encode_calls = []

    def encode_text(self, *, train_device, batch_size, text):
        self.encode_calls.append((train_device, batch_size, text))
        return (f"tensor-{text}", f"mask-{text}")


class GetPromptCacheTest(unittest.TestCase):
    def setUp(self):
        slider._PROMPT_CACHE = None

    def tearDown(self):
        slider._PROMPT_CACHE = None

    def test_memoized_across_repeated_calls(self):
        with mock.patch.object(slider_config, "TEXT_POSITIVE", "SENTINEL_POSITIVE"), \
             mock.patch.object(slider_config, "TEXT_NEUTRAL", "SENTINEL_NEUTRAL"), \
             mock.patch.object(slider_config, "TEXT_NEGATIVE", "SENTINEL_NEGATIVE"), \
             mock.patch.object(slider_config, "TEXT_TARGET", "SENTINEL_TARGET"):
            model = _StubModelForCache()
            train_device = torch.device("cpu")

            first = slider.get_prompt_cache(model, train_device)
            second = slider.get_prompt_cache(model, train_device)

        self.assertEqual(len(model.encode_calls), 4)
        self.assertIs(second, first)
        self.assertEqual(first.positive, ("tensor-SENTINEL_POSITIVE", "mask-SENTINEL_POSITIVE"))
        self.assertEqual(first.neutral, ("tensor-SENTINEL_NEUTRAL", "mask-SENTINEL_NEUTRAL"))
        self.assertEqual(first.negative, ("tensor-SENTINEL_NEGATIVE", "mask-SENTINEL_NEGATIVE"))
        self.assertEqual(first.target, ("tensor-SENTINEL_TARGET", "mask-SENTINEL_TARGET"))


class _Cfg:
    train_device = torch.device("cpu")

class _TP:
    global_step = 0


if __name__ == "__main__":
    unittest.main()
