from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp_tool import gate_detection


def valid_result(score, depth_points=200):
    return {
        "score": score,
        "box": (10, 10, 80, 80),
        "valid_depth_points": depth_points,
        "z_mm": 400.0,
    }


class DetectionGateTests(unittest.TestCase):
    def test_tape_measure_uses_wrist_specific_score_floor(self):
        gate = gate_detection(
            valid_result(0.42), "STABLE", "D435I",
            tool_category="tape measure")
        self.assertTrue(gate["passed"], gate)

    def test_other_tools_keep_shared_wrist_score_floor(self):
        gate = gate_detection(
            valid_result(0.42), "STABLE", "D435I",
            tool_category="screwdriver")
        self.assertFalse(gate["passed"])
        self.assertIn("score < 0.45", gate["reasons"])

    def test_fixed_camera_threshold_is_not_relaxed_for_tape(self):
        gate = gate_detection(
            valid_result(0.34), "STABLE", "D455",
            tool_category="tape measure")
        self.assertFalse(gate["passed"])
        self.assertIn("score < 0.35", gate["reasons"])

    def test_d455_screwdriver_accepts_valid_small_handle_core(self):
        gate = gate_detection(
            valid_result(0.50, depth_points=25), "STABLE", "D455",
            tool_category="screwdriver")
        self.assertTrue(gate["passed"], gate)

    def test_d455_screwdriver_still_requires_handle_depth(self):
        gate = gate_detection(
            valid_result(0.50, depth_points=19), "STABLE", "D455",
            tool_category="screwdriver")
        self.assertFalse(gate["passed"])
        self.assertIn("too few depth points (19<20)", gate["reasons"])

    def test_other_d455_tools_keep_eighty_point_floor(self):
        gate = gate_detection(
            valid_result(0.50, depth_points=79), "STABLE", "D455",
            tool_category="tape measure")
        self.assertFalse(gate["passed"])
        self.assertIn("too few depth points (79<80)", gate["reasons"])


if __name__ == "__main__":
    unittest.main()
