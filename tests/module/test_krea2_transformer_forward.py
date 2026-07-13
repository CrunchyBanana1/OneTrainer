import unittest

from modules.modelSetup.BaseKrea2Setup import BaseKrea2Setup

import torch


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


class _ConcreteKrea2Setup(BaseKrea2Setup):
    """BaseKrea2Setup is ABCMeta with unrelated abstract methods (create_parameters,
    setup_model, setup_train_device, after_optimizer_step) left unimplemented by
    BaseModelSetup. Python's ABC machinery blocks even __new__ on a class with
    unimplemented abstract methods, so this trivial subclass exists purely to allow
    bypassing __init__ (and its heavy deps) in the test below."""
    def create_parameters(self, *args, **kwargs):
        raise NotImplementedError

    def setup_model(self, *args, **kwargs):
        raise NotImplementedError

    def setup_train_device(self, *args, **kwargs):
        raise NotImplementedError

    def after_optimizer_step(self, *args, **kwargs):
        raise NotImplementedError


class Krea2TransformerForwardTest(unittest.TestCase):
    def _setup(self):
        # construct without running __init__ (avoids heavy deps); set train_device only
        setup = _ConcreteKrea2Setup.__new__(_ConcreteKrea2Setup)
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
