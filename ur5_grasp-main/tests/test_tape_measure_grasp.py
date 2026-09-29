from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from bsp.camera_bsp.planar_orientation import axial_difference_deg, principal_axis_base
from bsp.camera_bsp.tape_measure_grasp import (isolate_tape_measure_body,
                                                plan_tape_measure_grasp,
                                                select_tape_measure_body_result)


class TapeMeasureGeometryTests(unittest.TestCase):
    def test_body_midpoint_is_used_as_grasp_height(self):
        tcp_z, plan = plan_tape_measure_grasp(0.0614, 0.0214)
        self.assertIsNotNone(tcp_z, plan)
        self.assertAlmostEqual(plan["body_thickness_m"], 0.0400, places=4)
        self.assertAlmostEqual(plan["body_mid_z_m"], 0.0414, places=4)
        self.assertAlmostEqual(tcp_z, 0.0414, places=4)

    def test_tcp_offset_is_applied(self):
        tcp_z, plan = plan_tape_measure_grasp(0.0614, 0.0214, 0.004)
        self.assertIsNotNone(tcp_z, plan)
        self.assertAlmostEqual(tcp_z, 0.0454, places=4)

    def test_implausibly_thin_body_is_rejected(self):
        tcp_z, reason = plan_tape_measure_grasp(0.0300, 0.0214)
        self.assertIsNone(tcp_z)
        self.assertIn("small", reason)

    def test_implausibly_thick_body_is_rejected(self):
        tcp_z, reason = plan_tape_measure_grasp(0.1500, 0.0214)
        self.assertIsNone(tcp_z)
        self.assertIn("large", reason)

    def test_clearance_floor_is_enforced(self):
        tcp_z, plan = plan_tape_measure_grasp(
            0.0374, 0.0214, gripper_offset_m=-0.010,
            minimum_clearance_m=0.008)
        self.assertIsNotNone(tcp_z, plan)
        self.assertAlmostEqual(tcp_z, 0.0294, places=4)

    def test_typical_verified_tape_geometry_keeps_safe_clearance(self):
        tcp_z, plan = plan_tape_measure_grasp(
            0.0600, 0.0214, center_bias_m=-0.008)
        self.assertIsNotNone(tcp_z, plan)
        self.assertGreaterEqual(tcp_z - plan["support_plane_z_m"], 0.008)
        self.assertAlmostEqual(plan["body_thickness_m"], 0.0386, places=4)
        self.assertAlmostEqual(plan["applied_center_bias_m"], -0.008, places=4)
        self.assertAlmostEqual(tcp_z, 0.0327, places=4)

    def test_downward_bias_cannot_cross_support_clearance(self):
        tcp_z, plan = plan_tape_measure_grasp(
            0.0374, 0.0214, center_bias_m=-0.008,
            minimum_clearance_m=0.008)
        self.assertIsNotNone(tcp_z, plan)
        self.assertAlmostEqual(tcp_z, 0.0294, places=4)
        self.assertAlmostEqual(plan["applied_center_bias_m"], 0.0, places=4)

    def test_body_orientation_excludes_thin_tape_and_strap(self):
        mask = np.zeros((160, 220), dtype=bool)
        mask[45:115, 45:155] = True
        mask[76:84, 155:215] = True   # extended measuring tape
        mask[115:155, 96:104] = True  # wrist strap
        body, reason = isolate_tape_measure_body(mask)
        self.assertIsNone(reason)
        self.assertFalse(body[80, 190])
        self.assertFalse(body[140, 100])
        depth = np.full(mask.shape, 0.5)
        angle, reason = principal_axis_base(
            body, depth,
            lambda u, v, z: np.array([u * 0.001, v * 0.001, z]),
            minimum_eigenvalue_ratio=1.25)
        self.assertIsNone(reason)
        self.assertLess(axial_difference_deg(angle, 0.0), 2.0)

    def test_higher_score_strap_is_ignored_before_temporal_tracking(self):
        strap = np.zeros((160, 220), dtype=bool)
        strap[20:145, 94:108] = True
        case = np.zeros_like(strap)
        case[45:115, 45:155] = True
        case[76:84, 155:215] = True
        results = [
            {"mask": strap, "score": 0.70, "prompt": "a tape measure"},
            {"mask": case, "score": 0.45, "prompt": "a tape measure"},
        ]
        selected, reason = select_tape_measure_body_result(
            results, np.full(strap.shape, 500, np.uint16), 0.001)
        self.assertIsNone(reason)
        self.assertAlmostEqual(selected["score"], 0.45)
        self.assertGreater(selected["area"], 1000)
        self.assertFalse(selected["mask"][80, 190])

    def test_wide_strap_without_case_is_rejected(self):
        strap = np.zeros((160, 220), dtype=bool)
        strap[20:145, 94:108] = True
        body, reason = isolate_tape_measure_body(strap)
        self.assertIsNone(body)
        self.assertIn("strap-like", reason)


if __name__ == "__main__":
    unittest.main()
