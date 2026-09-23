import unittest

import numpy as np

from app import render
from segmentation.pipeline import (
    build_tool_catalog,
    classify_similarities,
    masked_object_crop,
    normalize_feature_rows,
    resolve_tool_request,
)


class SemanticRejectionTests(unittest.TestCase):
    def setUp(self):
        self.catalog = build_tool_catalog({
            "adjustable_wrench": {
                "localization_prompt": "adjustable wrench",
                "prompts": ["a photo of an adjustable wrench", "a wrench tool"],
                "aliases": ["wrench", "扳手"],
            },
            "screwdriver": {
                "localization_prompt": "screwdriver",
                "prompts": ["a photo of a screwdriver", "a screwdriver tool"],
                "aliases": ["screw driver", "螺丝刀"],
            },
            "pen": {
                "localization_prompt": "pen",
                "prompts": ["a photo of a pen", "a writing pen"],
                "aliases": ["笔"],
                "graspable": False,
            },
        })

    def test_resolves_chinese_and_phrase_aliases(self):
        self.assertEqual(resolve_tool_request("请给我扳手", self.catalog).name,
                         "adjustable_wrench")
        self.assertEqual(resolve_tool_request("grasp the screwdriver handle", self.catalog).name,
                         "screwdriver")

    def test_longest_matching_alias_wins(self):
        catalog = build_tool_catalog({
            "adjustable_wrench": {
                "localization_prompt": "adjustable wrench",
                "prompts": ["an adjustable wrench"],
                "aliases": ["wrench"],
            },
            "hex_key": {
                "localization_prompt": "hex key",
                "prompts": ["a hex key"],
                "aliases": ["hex wrench"],
            },
        })
        self.assertEqual(resolve_tool_request("please grasp the hex wrench", catalog).name,
                         "hex_key")

    def test_distractor_cannot_be_requested(self):
        with self.assertRaises(ValueError):
            resolve_tool_request("pen", self.catalog)

    def test_feature_rows_are_unit_normalized(self):
        result = normalize_feature_rows(np.array([[3.0, 4.0], [0.0, 2.0]], np.float32))
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), [1.0, 1.0], atol=1e-6)

    def test_zero_feature_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_feature_rows(np.zeros((1, 3), np.float32))

    def test_masked_crop_uses_neutral_background_and_padding(self):
        image = np.zeros((20, 20, 3), np.uint8)
        image[6:14, 8:12] = (10, 20, 30)
        mask = np.zeros((20, 20), bool)
        mask[6:14, 8:12] = True
        crop = np.asarray(masked_object_crop(image, mask, 0.25, 127))
        self.assertEqual(crop.shape, (12, 6, 3))
        self.assertTrue(np.all(crop[0, 0] == 127))
        self.assertTrue(np.any(np.all(crop == (10, 20, 30), axis=2)))

    def test_rejects_candidate_that_looks_more_like_screwdriver(self):
        scores, accepted, reason = classify_similarities({
            "adjustable_wrench": 0.24,
            "screwdriver": 0.31,
            "pen": 0.18,
        }, "adjustable_wrench", 0.20, 0.01)
        self.assertFalse(accepted)
        self.assertIn("screwdriver", reason)
        self.assertGreater(scores["screwdriver"], scores["adjustable_wrench"])

    def test_accepts_requested_class_with_clear_cosine_margin(self):
        _, accepted, reason = classify_similarities({
            "adjustable_wrench": 0.31,
            "screwdriver": 0.25,
            "pen": 0.17,
        }, "adjustable_wrench", 0.20, 0.01)
        self.assertTrue(accepted)
        self.assertIsNone(reason)

    def test_rejects_low_absolute_similarity(self):
        _, accepted, reason = classify_similarities({
            "adjustable_wrench": 0.18,
            "screwdriver": 0.15,
            "pen": 0.12,
        }, "adjustable_wrench", 0.20, 0.01)
        self.assertFalse(accepted)
        self.assertIn("cosine", reason)

    def test_rejects_ambiguous_cosine_margin(self):
        _, accepted, reason = classify_similarities({
            "adjustable_wrench": 0.251,
            "screwdriver": 0.247,
            "pen": 0.14,
        }, "adjustable_wrench", 0.20, 0.01)
        self.assertFalse(accepted)
        self.assertIn("margin", reason)

    def test_no_candidate_render_shows_red_rejection(self):
        image = np.zeros((120, 480, 3), np.uint8)
        rendered = render(image, [], None)
        self.assertTrue(np.any(rendered[:, :, 2] > 0))


if __name__ == "__main__":
    unittest.main()
