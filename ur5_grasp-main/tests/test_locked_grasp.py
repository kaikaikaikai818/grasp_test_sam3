from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp_tool import (POST_GRASP_LIFT_SPEED, SAFE_APPROACH_SPEED,
                        SAFE_DESCENT_SPEED, SAFE_DESCENT_TRANSIT_SPEED,
                        SCREWDRIVER_GRASP_SPEED, grasp_preview_status_text,
                        trusted_target_shift_m)


class LockedGraspTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
