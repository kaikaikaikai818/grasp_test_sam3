import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bsp.robot_bsp.tool_handover import (DirectionalPullDetector,
                                         execute_fixed_handover,
                                         load_handover_config)


LIMITS = [[-0.5, 0.2], [-0.8, -0.3], [0.0, 0.7]]


def approved_file_config():
    return {
        "approved_for_this_cell": True,
        "handover_xyz_m": [-0.1, -0.5, 0.20],
        "travel_tcp_z_m": 0.35,
        "pull_direction_base_xy": [1.0, 0.0],
        "pull_threshold_n": 8.0,
        "max_lateral_force_n": 10.0,
        "pull_sustain_s": 0.25,
        "baseline_duration_s": 1.0,
        "handover_timeout_s": 20.0,
        "retreat_distance_m": 0.08,
        "force_calibration": {
            "light_touch_peak_n": [3.0, 4.0, 5.0],
            "normal_take_peak_n": [10.0, 11.0, 12.0],
            "normal_take_duration_s": [0.4, 0.5, 0.6],
        },
        "tools": {
            "screwdriver": {
                "approved": True,
                "orientation_rvec": [3.141, 0.0, 0.0],
            }
        },
    }


def runtime_config():
    return {
        "category": "screwdriver",
        "handover_pose": [-0.1, -0.5, 0.20, 3.141, 0.0, 0.0],
        "travel_tcp_z_m": 0.35,
        "pull_direction_base_xy": [1.0, 0.0],
        "pull_threshold_n": 8.0,
        "max_lateral_force_n": 10.0,
        "pull_sustain_s": 0.25,
        "baseline_duration_s": 1.0,
        "handover_timeout_s": 20.0,
        "retreat_distance_m": 0.08,
    }


class FakeRobot:
    def __init__(self):
        self.pose = [-0.2, -0.6, 0.25, 3.141, 0.0, 0.0]
        self.moves = []
        self.releases = 0

    def get_actual_tcp_pose(self):
        return self.pose

    def moveL(self, pose, **kwargs):
        self.pose = list(pose)
        self.moves.append(list(pose))

    def grip(self, *args):
        self.releases += 1
        return 4000


class HandoverTests(unittest.TestCase):
    def test_unapproved_configuration_cannot_move_robot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "handover.json"
            path.write_text(json.dumps({"approved_for_this_cell": False}),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not approved"):
                load_handover_config(path, "screwdriver", LIMITS)

    def test_directional_detector_rejects_touch_and_side_load(self):
        detector = DirectionalPullDetector([0] * 6, [1, 0], 8, 10, 0.25)
        self.assertFalse(detector.update([4, 0, 0, 0, 0, 0], 0.0)[0])
        self.assertFalse(detector.update([9, 15, 0, 0, 0, 0], 0.1)[0])

    def test_directional_detector_rejects_zero_direction(self):
        with self.assertRaisesRegex(ValueError, "non-zero"):
            DirectionalPullDetector([0] * 6, [0, 0], 8, 10, 0.25)

    def test_approved_configuration_loads_tool_specific_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "handover.json"
            path.write_text(json.dumps(approved_file_config()), encoding="utf-8")
            config = load_handover_config(path, "screwdriver", LIMITS)
        self.assertEqual(config["handover_pose"][:3], [-0.1, -0.5, 0.2])
        self.assertEqual(config["pull_direction_base_xy"], [1.0, 0.0])

    def test_threshold_must_be_supported_by_six_calibration_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "handover.json"
            data = approved_file_config()
            data["pull_threshold_n"] = 4.5
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not separate"):
                load_handover_config(path, "screwdriver", LIMITS)

    def test_directional_detector_requires_sustained_outward_pull(self):
        detector = DirectionalPullDetector([0] * 6, [1, 0], 8, 10, 0.25)
        self.assertFalse(detector.update([9, 1, 0, 0, 0, 0], 1.0)[0])
        self.assertFalse(detector.update([10, 1, 0, 0, 0, 0], 1.2)[0])
        self.assertTrue(detector.update([10, 1, 0, 0, 0, 0], 1.26)[0])

    def test_timeout_keeps_gripper_closed_and_robot_at_handover(self):
        config = runtime_config()
        robot = FakeRobot()
        ok, reason = execute_fixed_handover(
            robot, config, LIMITS, 4000,
            wait_for_pull=lambda *_: (False, "timeout"))
        self.assertFalse(ok)
        self.assertEqual(reason, "timeout")
        self.assertEqual(robot.releases, 0)
        self.assertEqual(robot.pose, config["handover_pose"])

    def test_confirmed_pull_releases_then_retreats(self):
        config = runtime_config()
        robot = FakeRobot()
        ok, reason = execute_fixed_handover(
            robot, config, LIMITS, 4000,
            wait_for_pull=lambda *_: (True, None))
        self.assertTrue(ok, reason)
        self.assertEqual(robot.releases, 1)
        self.assertEqual(len(robot.moves), 5)
        self.assertLess(robot.moves[-1][0], config["handover_pose"][0])
        self.assertEqual(robot.moves[-1][2], config["travel_tcp_z_m"])


if __name__ == "__main__":
    unittest.main()
