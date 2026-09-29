from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import grasp_tool
from grasp_tool import (POST_GRASP_LIFT_SPEED, SAFE_APPROACH_SPEED,
                        SAFE_DESCENT_SPEED, SAFE_DESCENT_TRANSIT_SPEED,
                        SCREWDRIVER_GRASP_SPEED, clearance_meets_minimum,
                        grasp_preview_status_text,
                        trusted_target_shift_m)


class LockedGraspTests(unittest.TestCase):
    def test_exact_three_mm_clearance_survives_float_subtraction(self):
        clearance = 0.0244 - 0.0214
        self.assertTrue(clearance_meets_minimum(clearance, 0.003))
        self.assertFalse(clearance_meets_minimum(0.0029, 0.003))

    def test_occluded_low_view_does_not_replace_locked_target(self):
        shift = trusted_target_shift_m(
            [0.0, -0.6, 0.05], None,
            {"passed": False, "reasons": ["not stable"]},
            live_region_available=False)
        self.assertIsNone(shift)

    def test_trusted_low_view_still_checks_target_motion(self):
        shift = trusted_target_shift_m(
            [0.0, -0.6, 0.05], [0.02, -0.6, 0.05],
            {"passed": True, "reasons": []})
        self.assertAlmostEqual(shift, 0.02)

    def test_completed_descent_status_does_not_depend_on_live_preview(self):
        text = grasp_preview_status_text(
            {"ready": False, "reason": "waiting for stable D435i target"},
            safe_descent_enabled=True, safe_descent_completed=True)
        self.assertIn("locked plan ready", text)

    def test_motion_speed_profile_slows_down_toward_the_tool(self):
        self.assertEqual(SAFE_APPROACH_SPEED, 0.20)
        self.assertGreater(SAFE_APPROACH_SPEED, SAFE_DESCENT_TRANSIT_SPEED)
        self.assertGreater(SAFE_DESCENT_TRANSIT_SPEED, SAFE_DESCENT_SPEED)
        self.assertGreater(SAFE_DESCENT_SPEED, SCREWDRIVER_GRASP_SPEED)
        self.assertLessEqual(POST_GRASP_LIFT_SPEED, SAFE_DESCENT_TRANSIT_SPEED)

    def test_descent_has_no_physical_100mm_pregrasp_stop(self):
        class FakeRobot:
            def __init__(self):
                self.pose = np.array([0.0, -0.6, 0.25, *grasp_tool.TOOL_ORIENTATION])
                self.moves = []
                self.rtde_c = None

            def get_actual_tcp_pose(self):
                return self.pose.tolist()

            def moveL(self, pose, speed, acceleration):
                self.pose = np.asarray(pose, dtype=np.float64)
                self.moves.append((self.pose.copy(), speed, acceleration))

        robot = FakeRobot()
        preview = {
            "ready": True,
            "target_surface_xyz_m": [0.01, -0.61, 0.05],
            "endpoint_xyz_m": [0.01, -0.61, 0.034],
            "pregrasp_xyz_m": [0.01, -0.61, 0.134],
            "orientation": list(grasp_tool.TOOL_ORIENTATION),
        }
        with patch.object(grasp_tool, "verify_tool_orientation", return_value=True):
            self.assertTrue(grasp_tool.move_to_safe_descent_test(robot, preview))

        self.assertEqual(len(robot.moves), 2)
        moved_z = [round(float(move[0][2]), 3) for move in robot.moves]
        self.assertNotIn(0.134, moved_z)
        self.assertAlmostEqual(moved_z[-1], 0.074, places=3)


if __name__ == "__main__":
    unittest.main()
