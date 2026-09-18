from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bsp.camera_bsp.screwdriver_grasp import (find_screwdriver_handle,
                                               fit_horizontal_support_plane)


class ScrewdriverGeometryTests(unittest.TestCase):
    def test_thick_handle_wins_over_thin_shaft(self):
        mask = np.zeros((100, 160), dtype=np.uint8)
        mask[45:55, 15:105] = 1        # shaft
        mask[28:72, 104:145] = 1       # handle
        depth = np.full(mask.shape, 500, dtype=np.uint16)
        handle, reason = find_screwdriver_handle({"mask": mask.astype(bool)}, depth, 0.001)
        self.assertIsNotNone(handle, reason)
        self.assertGreater(handle.center_px[0], 110)

    def test_narrow_mask_is_rejected(self):
        mask = np.zeros((80, 120), dtype=bool)
        mask[38:42, 10:110] = True
        handle, reason = find_screwdriver_handle({"mask": mask}, np.full(mask.shape, 500, np.uint16), 0.001)
        self.assertIsNone(handle)
        self.assertIn("narrow", reason)

    def test_dominant_horizontal_plane_is_fitted(self):
        depth = np.full((60, 80), 500, dtype=np.uint16)
        plane, reason = fit_horizontal_support_plane(
            depth, 0.001, lambda u, v, z: np.array([u / 1000.0, v / 1000.0, 0.123]), stride=4)
        self.assertIsNotNone(plane, reason)
        self.assertAlmostEqual(plane.z_m, 0.123, places=4)


if __name__ == "__main__":
    unittest.main()
