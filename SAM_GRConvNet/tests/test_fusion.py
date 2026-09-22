import unittest

import numpy as np

from grasp_backend.fusion import select_masked_grasp


class FusionTests(unittest.TestCase):
    def maps(self):
        return {
            "quality": np.array([[0.9, 0.2], [0.3, 0.4]]),
            "angle": np.zeros((2, 2)),
            "width": np.full((2, 2), 1.0),
            "angle_strength": np.ones((2, 2)),
        }

    def test_background_peak_cannot_win(self):
        maps = {key: np.pad(value, 10, mode="edge") for key, value in self.maps().items()}
        mask = np.ones((22, 22), bool)
        mask[:11, :11] = False
        result = select_masked_grasp(maps, mask, min_quality=0)
        self.assertTrue(result.accepted)
        self.assertNotEqual(result.candidate.center_xy, (0, 0))

    def test_small_mask_is_rejected(self):
        result = select_masked_grasp(self.maps(), np.ones((2, 2), bool))
        self.assertFalse(result.accepted)
        self.assertIn("small", result.rejection_reason)

    def test_weak_angle_is_rejected(self):
        shape = (20, 20)
        maps = {"quality": np.ones(shape), "angle": np.zeros(shape),
                "width": np.full(shape, 4.0), "angle_strength": np.zeros(shape)}
        result = select_masked_grasp(maps, np.ones(shape, bool))
        self.assertFalse(result.accepted)
        self.assertIn("angle", result.rejection_reason)

    def test_excessive_width_is_rejected(self):
        shape = (20, 20)
        maps = {"quality": np.ones(shape), "angle": np.zeros(shape),
                "width": np.full(shape, 30.0), "angle_strength": np.ones(shape)}
        result = select_masked_grasp(maps, np.ones(shape, bool))
        self.assertFalse(result.accepted)
        self.assertIn("width", result.rejection_reason)

    def test_width_is_checked_at_candidate_location(self):
        shape = (60, 120)
        mask = np.zeros(shape, bool)
        mask[25:35, 10:110] = True
        maps = {"quality": np.zeros(shape), "angle": np.zeros(shape),
                "width": np.full(shape, 30.0), "angle_strength": np.ones(shape)}
        maps["quality"][30, 60] = 1.0
        result = select_masked_grasp(maps, mask, max_width_to_minor_bbox=10.0)
        self.assertFalse(result.accepted)
        self.assertIn("local target width", result.rejection_reason)


if __name__ == "__main__":
    unittest.main()
