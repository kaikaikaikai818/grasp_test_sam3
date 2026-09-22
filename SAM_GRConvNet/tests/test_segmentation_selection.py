import unittest

import numpy as np

from segmentation.pipeline import select_complete_mask


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


if __name__ == "__main__":
    unittest.main()
