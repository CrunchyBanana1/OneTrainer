import unittest

from modules.trainer.slider import compose_text_target, image_multiplier_for_path

import torch


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
