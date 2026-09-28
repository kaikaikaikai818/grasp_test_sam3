from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp_tool import gate_detection


def valid_result(score):
    return {
        "score": score,
        "box": (10, 10, 80, 80),
        "valid_depth_points": 200,
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


if __name__ == "__main__":
    unittest.main()
