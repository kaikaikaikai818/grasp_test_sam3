from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bsp.camera_bsp.screwdriver_grasp import (estimate_handle_thickness,
                                               find_screwdriver_handle,
                                               fit_horizontal_support_plane,
                                               plan_grasp_tcp)


class ScrewdriverGeometryTests(unittest.TestCase):
    def test_handle_diameter_from_projected_width(self):
        # A 30 mm cylinder at 0.25 m with f=600 px projects to a 36 px radius.
        thickness, reason = estimate_handle_thickness(36.0, 0.25, 600.0)
        self.assertIsNotNone(thickness, reason)
        self.assertAlmostEqual(thickness, 0.030, places=4)

    def test_plan_grasp_tcp_uses_mid_and_offset(self):
        tcp_z, plan = plan_grasp_tcp(0.050, 0.000, 0.0047)
        self.assertIsNotNone(tcp_z, plan)
        self.assertAlmostEqual(tcp_z, 0.0297, places=4)

    def test_plan_grasp_tcp_rejects_implausible_thickness(self):
        tcp_z, reason = plan_grasp_tcp(0.001, 0.000, 0.0)
        self.assertIsNone(tcp_z)
        self.assertIn("small", reason)

    def test_thick_handle_wins_over_thin_shaft(self):
        mask = np.zeros((100, 160), dtype=np.uint8)
        mask[45:55, 15:105] = 1        # shaft
        mask[28:72, 104:145] = 1       # handle
        depth = np.full(mask.shape, 500, dtype=np.uint16)
        handle, reason = find_screwdriver_handle({"mask": mask.astype(bool)}, depth, 0.001)
        self.assertIsNotNone(handle, reason)
        self.assertIsNone(reason)
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
