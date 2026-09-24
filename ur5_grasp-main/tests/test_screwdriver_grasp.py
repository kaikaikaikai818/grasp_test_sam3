from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bsp.camera_bsp.screwdriver_grasp import (base_point_m,
                                               estimate_handle_thickness,
                                               find_screwdriver_handle,
                                               fit_horizontal_support_plane,
                                               locked_support_plane_z,
                                               load_calibration,
                                               plan_grasp_tcp)
from bsp.camera_bsp.screwdriver_grasp import SupportPlane, verified_support_plane_z


class ScrewdriverGeometryTests(unittest.TestCase):
    def test_camera_to_base_metre_value_is_not_scaled_twice(self):
        transform_result = (np.array([100.0, -200.0, 19.0]),
                            np.array([0.100, -0.200, 0.019]))
        np.testing.assert_allclose(
            base_point_m(transform_result), [0.100, -0.200, 0.019])

    def test_invalid_camera_transform_result_is_rejected(self):
        with self.assertRaises(ValueError):
            base_point_m(np.array([0.0, 0.0, 0.019]))

    def test_plane_only_calibration_uses_tcp_clamp_center(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plane.json"
            path.write_text(json.dumps({
                "support_plane_z_m": 0.0215,
                "support_plane_spread_m": 0.0055,
            }), encoding="utf-8")
            calibration = load_calibration(path, default_gripper_offset_m=0.0)
        self.assertEqual(calibration["gripper_offset_m"], 0.0)
        self.assertEqual(calibration["gripper_offset_source"], "active_tcp_clamp")

    def test_tcp_clamp_center_overrides_legacy_contact_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plane.json"
            path.write_text(json.dumps({
                "support_plane_z_m": 0.0215,
                "support_plane_spread_m": 0.0055,
                "gripper_offset_m": 0.012,
            }), encoding="utf-8")
            calibration = load_calibration(path, default_gripper_offset_m=0.0)
        self.assertEqual(calibration["gripper_offset_m"], 0.0)
        self.assertEqual(calibration["gripper_offset_source"], "active_tcp_clamp")

    def test_saved_fixed_plane_requires_live_agreement(self):
        calibration = {"support_plane_z_m": 0.0215}
        accepted, reason = verified_support_plane_z(
            SupportPlane(0.0240, 0.001, 100), calibration)
        self.assertAlmostEqual(accepted, 0.0215)
        self.assertIsNone(reason)
        rejected, reason = verified_support_plane_z(
            SupportPlane(0.0400, 0.001, 100), calibration)
        self.assertIsNone(rejected)
        self.assertIn("differs", reason)

    def test_handle_diameter_from_projected_width(self):
        # A 30 mm cylinder at 0.25 m with f=600 px projects to a 36 px radius.
        thickness, reason = estimate_handle_thickness(36.0, 0.25, 600.0)
        self.assertIsNotNone(thickness, reason)
        self.assertAlmostEqual(thickness, 0.030, places=4)

    def test_plan_grasp_tcp_uses_mid_and_offset(self):
        tcp_z, plan = plan_grasp_tcp(0.050, 0.0215, 0.0047)
        self.assertIsNotNone(tcp_z, plan)
        self.assertAlmostEqual(tcp_z, 0.0512, places=4)
        self.assertAlmostEqual(plan["support_plane_z_m"], 0.0215, places=4)

    def test_descent_locks_saved_plane_before_motion(self):
        locked, reason = locked_support_plane_z(
            {"support_plane_z_m": 0.0214},
            calibration={"support_plane_z_m": 0.0215},
            live_plane=SupportPlane(0.0212, 0.001, 100))
        self.assertIsNone(reason)
        self.assertAlmostEqual(locked, 0.0215, places=4)

    def test_descent_rejects_missing_or_mismatched_planned_plane(self):
        locked, reason = locked_support_plane_z(
            {}, calibration={"support_plane_z_m": 0.0215})
        self.assertIsNone(locked)
        self.assertIn("plan", reason)
        locked, reason = locked_support_plane_z(
            {"support_plane_z_m": 0.0180},
            calibration={"support_plane_z_m": 0.0215})
        self.assertIsNone(locked)
        self.assertIn("differ", reason)

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
