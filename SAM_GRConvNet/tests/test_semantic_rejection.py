import unittest

import numpy as np

from segmentation.pipeline import (
    build_tool_catalog, classify_mask, resolve_tool_request,
)


class SemanticRejectionTests(unittest.TestCase):
    def setUp(self):
        self.catalog = build_tool_catalog({
            "adjustable_wrench": {
                "prompt": "adjustable wrench", "aliases": ["wrench", "扳手"],
            },
            "screwdriver": {
                "prompt": "screwdriver", "aliases": ["screw driver", "螺丝刀"],
            },
            "pen": {"prompt": "pen", "aliases": ["笔"], "graspable": False},
        })

    def test_resolves_chinese_and_phrase_aliases(self):
        self.assertEqual(resolve_tool_request("请给我扳手", self.catalog).name,
                         "adjustable_wrench")
        self.assertEqual(resolve_tool_request("grasp the screwdriver handle", self.catalog).name,
                         "screwdriver")

    def test_longest_matching_alias_wins(self):
        catalog = build_tool_catalog({
            "adjustable_wrench": {"prompt": "adjustable wrench", "aliases": ["wrench"]},
            "hex_key": {"prompt": "hex key", "aliases": ["hex wrench"]},
        })
        self.assertEqual(resolve_tool_request("please grasp the hex wrench", catalog).name,
                         "hex_key")

    def test_distractor_cannot_be_requested(self):
        with self.assertRaises(ValueError):
            resolve_tool_request("pen", self.catalog)

    def test_rejects_mask_that_looks_more_like_screwdriver(self):
        mask = np.ones((10, 10), dtype=bool)
        maps = {
            "adjustable_wrench": np.full((10, 10), 0.55),
            "screwdriver": np.full((10, 10), 0.82),
            "pen": np.full((10, 10), 0.20),
        }
        scores, accepted, reason = classify_mask(
            maps, mask, "adjustable_wrench", 0.35, 0.03, 0.25,
        )
        self.assertFalse(accepted)
        self.assertIn("screwdriver", reason)
        self.assertGreater(scores["screwdriver"], scores["adjustable_wrench"])

    def test_accepts_requested_class_with_clear_margin(self):
        mask = np.ones((10, 10), dtype=bool)
        maps = {
            "adjustable_wrench": np.full((10, 10), 0.81),
            "screwdriver": np.full((10, 10), 0.53),
            "pen": np.full((10, 10), 0.31),
        }
        _, accepted, reason = classify_mask(
            maps, mask, "adjustable_wrench", 0.35, 0.03, 0.25,
        )
        self.assertTrue(accepted)
        self.assertIsNone(reason)

    def test_rejects_ambiguous_margin(self):
        mask = np.ones((10, 10), dtype=bool)
        maps = {
            "adjustable_wrench": np.full((10, 10), 0.61),
            "screwdriver": np.full((10, 10), 0.60),
            "pen": np.full((10, 10), 0.20),
        }
        _, accepted, reason = classify_mask(
            maps, mask, "adjustable_wrench", 0.35, 0.03, 0.25,
        )
        self.assertFalse(accepted)
        self.assertIn("margin", reason)


if __name__ == "__main__":
    unittest.main()
