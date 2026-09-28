from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bsp.camera_bsp.tape_measure_grasp import plan_tape_measure_grasp


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
        tcp_z, plan = plan_tape_measure_grasp(0.0600, 0.0214)
        self.assertIsNotNone(tcp_z, plan)
        self.assertGreaterEqual(tcp_z - plan["support_plane_z_m"], 0.008)
        self.assertAlmostEqual(plan["body_thickness_m"], 0.0386, places=4)


if __name__ == "__main__":
    unittest.main()
