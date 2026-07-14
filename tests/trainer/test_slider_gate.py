# tests/trainer/test_slider_gate.py
import unittest
from unittest import mock

from modules.trainer import slider_config
from modules.trainer.slider import slider_enabled


class SliderEnabledTest(unittest.TestCase):
    def test_disabled_when_none(self):
        with mock.patch.object(slider_config, "SLIDER_MODE", None):
            self.assertFalse(slider_enabled())

    def test_enabled_for_text(self):
        with mock.patch.object(slider_config, "SLIDER_MODE", "text"):
            self.assertTrue(slider_enabled())


if __name__ == "__main__":
    unittest.main()
