import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bsp.camera_bsp.planar_orientation import (axial_difference_deg,
                                               overhead_orientation,
                                               principal_axis_base)


class PlanarOrientationTests(unittest.TestCase):
    def test_camera_rotation_does_not_change_base_axis(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[46:54, 20:80] = True
        depth = np.full(mask.shape, 0.6)
        def rotated(u, v, z):
            return np.array([-v * 0.001, u * 0.001, z])
        angle, reason = principal_axis_base(mask, depth, rotated)
        self.assertIsNone(reason)
        self.assertLess(axial_difference_deg(angle, np.pi / 2), 2.0)

    def test_round_or_invalid_mask_is_rejected(self):
        mask = np.zeros((80, 80), dtype=bool)
        mask[25:55, 25:55] = True
        angle, reason = principal_axis_base(
            mask, np.full(mask.shape, 0.5),
            lambda u, v, z: np.array([u * .001, v * .001, z]))
        self.assertIsNone(angle)
        self.assertIn("long axis", reason)

    def test_closing_axis_is_perpendicular_and_tilt_remains_down(self):
        pose = overhead_orientation(0.0, [np.pi, 0, 0])
        target, _ = cv2.Rodrigues(np.asarray(pose))
        self.assertAlmostEqual(abs(target[0, 0]), 0.0, places=5)
        self.assertAlmostEqual(target[2, 2], -1.0, places=5)


if __name__ == "__main__":
    unittest.main()
