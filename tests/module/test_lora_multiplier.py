import unittest

from modules.module.LoRAModule import LoRAModule, LoRAModuleWrapper
from modules.util.enum.ModelType import PeftType

import torch
from torch import nn


def _make_lora_over_linear(rank=4, alpha=4.0, in_f=8, out_f=8):
    orig = nn.Linear(in_f, out_f, bias=False)
    module = LoRAModule("test", orig, rank, alpha)
    # check_initialized() asserts orig_forward is set, which only happens once the
    # module is hooked onto its original module; delta_forward() is normally only
    # reached via forward() after hooking, so replicate that here for direct calls.
    module.hook_to_module()
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
    # LoRAModuleWrapper.__init__ (LoRAModule.py:838+) also reads these regardless of
    # module_filter being empty: peft_type selects the LORA branch, which in turn
    # reads lora_decompose; lokr_dim is read unconditionally before the branch.
    peft_type = PeftType.LORA
    lokr_dim = 4
    lora_decompose = False


if __name__ == "__main__":
    unittest.main()
