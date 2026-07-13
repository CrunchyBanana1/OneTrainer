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
