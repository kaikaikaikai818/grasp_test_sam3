import unittest

import numpy as np

from segmentation.pipeline import SamObject, merge_overlapping_objects, select_complete_mask


class CompleteMaskSelectionTests(unittest.TestCase):
    def test_prefers_complete_object_with_similar_sam_score(self):
        masks = np.zeros((3, 20, 20), dtype=bool)
        masks[0, 4:8, 8:12] = True
        masks[1, 4:17, 7:13] = True
        masks[2, :, :] = True
        scores = np.array([0.94, 0.88, 0.92])

        selected, score = select_complete_mask(masks, scores, (9, 5))

        np.testing.assert_array_equal(selected, masks[1])
        self.assertAlmostEqual(score, 0.88)

    def test_rejects_large_background_proposal(self):
        masks = np.zeros((2, 20, 20), dtype=bool)
        masks[0, 3:18, 7:13] = True
        masks[1, :, :] = True
        scores = np.array([0.80, 0.99])

        selected, _ = select_complete_mask(masks, scores, (9, 5))

        np.testing.assert_array_equal(selected, masks[0])

    def test_merges_overlapping_part_and_whole_proposals(self):
        whole = np.zeros((30, 20), dtype=bool)
        whole[3:27, 7:13] = True
        part = np.zeros_like(whole)
        part[3:10, 6:14] = True
        objects = [
            SamObject(whole, (7, 3, 13, 27), 0.7, 0.8),
            SamObject(part, (6, 3, 14, 10), 0.8, 0.9),
        ]

        merged = merge_overlapping_objects(objects)

        self.assertEqual(len(merged), 1)
        np.testing.assert_array_equal(merged[0].mask, np.logical_or(whole, part))

    def test_keeps_separate_non_overlapping_tools(self):
        left = np.zeros((20, 30), dtype=bool)
        right = np.zeros_like(left)
        left[4:16, 2:8] = True
        right[4:16, 22:28] = True

        merged = merge_overlapping_objects([
            SamObject(left, (2, 4, 8, 16), 0.8, 0.9),
            SamObject(right, (22, 4, 28, 16), 0.8, 0.9),
        ])

        self.assertEqual(len(merged), 2)


if __name__ == "__main__":
    unittest.main()
