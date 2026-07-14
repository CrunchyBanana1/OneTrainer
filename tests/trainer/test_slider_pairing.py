import unittest

from modules.trainer import slider

import torch


def _batch(path, shape=(4, 8, 8)):
    # a minimal batch mirroring what the data loader yields: batch_size=1
    return {'image_path': [path], 'latent_image': [torch.zeros(*shape)]}


class BuildImageSliderBatchesTest(unittest.TestCase):
    def test_pairs_positive_with_negative_twin_adjacently(self):
        pos = _batch("/data/positive/a.png")
        neg = _batch("/data/negative/a.png")
        # deliberately not already adjacent / in order
        paired, dropped = slider.build_image_slider_batches([neg, pos], "positive", "negative")
        self.assertEqual(dropped, [])
        self.assertEqual([b['image_path'][0] for b in paired],
                         ["/data/positive/a.png", "/data/negative/a.png"])

    def test_multiple_pairs_stay_grouped(self):
        batches = [
            _batch("/d/positive/a.png"), _batch("/d/positive/b.png"),
            _batch("/d/negative/a.png"), _batch("/d/negative/b.png"),
        ]
        paired, dropped = slider.build_image_slider_batches(batches, "positive", "negative")
        self.assertEqual(dropped, [])
        self.assertEqual([b['image_path'][0] for b in paired],
                         ["/d/positive/a.png", "/d/negative/a.png",
                          "/d/positive/b.png", "/d/negative/b.png"])

    def test_drops_positive_without_twin(self):
        pos = _batch("/data/positive/lonely.png")
        paired, dropped = slider.build_image_slider_batches([pos], "positive", "negative")
        self.assertEqual(paired, [])
        self.assertEqual(dropped, ["/data/positive/lonely.png"])

    def test_drops_on_latent_shape_mismatch(self):
        pos = _batch("/d/positive/a.png", shape=(4, 8, 8))
        neg = _batch("/d/negative/a.png", shape=(4, 16, 16))
        paired, dropped = slider.build_image_slider_batches([pos, neg], "positive", "negative")
        self.assertEqual(paired, [])
        self.assertEqual(dropped, ["/d/positive/a.png"])

    def test_standalone_negative_is_not_emitted(self):
        neg = _batch("/data/negative/orphan.png")
        paired, dropped = slider.build_image_slider_batches([neg], "positive", "negative")
        self.assertEqual(paired, [])
        self.assertEqual(dropped, [])

    def test_raises_on_batch_size_greater_than_one(self):
        multi = {'image_path': ["/d/positive/a.png", "/d/positive/b.png"],
                 'latent_image': [torch.zeros(4, 8, 8), torch.zeros(4, 8, 8)]}
        with self.assertRaises(ValueError):
            slider.build_image_slider_batches([multi], "positive", "negative")

    def test_twins_get_same_stamped_seed_varying_per_pair(self):
        batches = [
            _batch("/d/positive/a.png"), _batch("/d/positive/b.png"),
            _batch("/d/negative/a.png"), _batch("/d/negative/b.png"),
        ]
        paired, _ = slider.build_image_slider_batches(batches, "positive", "negative", base_seed=100)
        # [posA, negA, posB, negB] -> pair A seed 100 (both), pair B seed 101 (both)
        seeds = [b['slider_pair_seed'] for b in paired]
        self.assertEqual(seeds, [100, 100, 101, 101])

    def test_base_seed_shifts_all_pair_seeds(self):
        batches = [_batch("/d/positive/a.png"), _batch("/d/negative/a.png")]
        paired, _ = slider.build_image_slider_batches(batches, "positive", "negative", base_seed=500)
        self.assertEqual([b['slider_pair_seed'] for b in paired], [500, 500])

    def test_defaults_to_config_tokens(self):
        # tokens omitted -> falls back to slider_config defaults ("positive"/"negative")
        pos = _batch("/data/positive/a.png")
        neg = _batch("/data/negative/a.png")
        paired, dropped = slider.build_image_slider_batches([pos, neg])
        self.assertEqual(len(paired), 2)
        self.assertEqual(dropped, [])


if __name__ == "__main__":
    unittest.main()
