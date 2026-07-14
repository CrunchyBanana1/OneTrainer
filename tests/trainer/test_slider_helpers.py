import unittest

from modules.trainer.slider import compose_text_target

import torch


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
