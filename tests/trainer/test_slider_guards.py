# tests/trainer/test_slider_guards.py
import unittest

from modules.module.quantized.LinearSVD import BaseLinearSVD
from modules.trainer import slider

import torch
from torch import nn


class _ConcreteBaseLinearSVD(BaseLinearSVD):
    """Minimal concrete BaseLinearSVD: implements just enough of the abstract mixins to
    instantiate a real `isinstance(..., BaseLinearSVD)` object, without any of the real
    quantization machinery (which needs a lot more setup than a unit test wants)."""
    def quantize(self, device=None):
        pass

    def original_weight_shape(self):
        return (1, 1)

    def unquantized_weight(self, dtype, device):
        return torch.zeros(1, 1, dtype=dtype, device=device)

    def forward_with_lora(self, x, lora_down, lora_up, dropout, alpha):
        return x


class _StubWrappedModule:
    """Stands in for a PeftBase-like wrapped module: exposes only `orig_module`."""
    def __init__(self, orig_module):
        self.orig_module = orig_module


class _StubTransformerLora:
    def __init__(self, lora_modules):
        self.lora_modules = lora_modules


class _StubModel:
    def __init__(self, transformer_lora):
        self.transformer_lora = transformer_lora


class CheckSliderCompatibleTest(unittest.TestCase):
    def test_raises_when_transformer_lora_missing(self):
        model = _StubModel(transformer_lora=None)
        with self.assertRaises(RuntimeError):
            slider.check_slider_compatible(model)

    def test_raises_when_transformer_lora_attribute_absent(self):
        model = object()  # no transformer_lora attribute at all
        with self.assertRaises(RuntimeError):
            slider.check_slider_compatible(model)

    def test_raises_when_any_module_wraps_quantized_linear(self):
        quantized = _ConcreteBaseLinearSVD()
        lora_modules = {
            "a": _StubWrappedModule(nn.Linear(4, 4)),
            "b": _StubWrappedModule(quantized),
        }
        model = _StubModel(transformer_lora=_StubTransformerLora(lora_modules))
        with self.assertRaises(RuntimeError):
            slider.check_slider_compatible(model)

    def test_passes_for_plain_linear_modules(self):
        lora_modules = {
            "a": _StubWrappedModule(nn.Linear(4, 4)),
            "b": _StubWrappedModule(nn.Linear(4, 4)),
        }
        model = _StubModel(transformer_lora=_StubTransformerLora(lora_modules))
        slider.check_slider_compatible(model)  # must not raise

    def test_skips_modules_whose_orig_module_is_none(self):
        # dummy/placeholder wrapped modules (orig_module=None) must not blow up the check.
        lora_modules = {
            "a": _StubWrappedModule(None),
            "b": _StubWrappedModule(nn.Linear(4, 4)),
        }
        model = _StubModel(transformer_lora=_StubTransformerLora(lora_modules))
        slider.check_slider_compatible(model)  # must not raise


class ResetPromptCacheTest(unittest.TestCase):
    def test_sets_prompt_cache_to_none(self):
        slider._PROMPT_CACHE = slider.SliderPromptCache(
            positive=("p",), neutral=("n",), negative=("g",), target=("t",),
        )
        slider.reset_prompt_cache()
        self.assertIsNone(slider._PROMPT_CACHE)

    def tearDown(self):
        slider._PROMPT_CACHE = None


if __name__ == "__main__":
    unittest.main()
