from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp_tool import associate_targets, safe_approach_candidate


PASS_GATE = {"passed": True, "reasons": []}


class TargetAssociationTests(unittest.TestCase):
    def test_screwdriver_allows_center_shift_along_tool_axis(self):
        axis = np.deg2rad(130.6)
        direction = np.array([np.cos(axis), np.sin(axis)])
        lateral = np.array([-direction[1], direction[0]])
        ho = np.array([-0.081, -0.593, 0.049])
        hi = ho.copy()
        hi[:2] += direction * 0.021 + lateral * 0.003
        hi[2] += 0.001
        association = associate_targets(
            ho, hi, PASS_GATE, PASS_GATE, alignment_applied=True,
            tool_category="screwdriver", tool_axis_rad=axis)
        self.assertTrue(association["safe_approach_matched"], association)
        destination, reason = safe_approach_candidate(association)
        self.assertIsNotNone(destination, reason)

    def test_screwdriver_keeps_cross_axis_limit_strict(self):
        axis = np.deg2rad(130.6)
        direction = np.array([np.cos(axis), np.sin(axis)])
        lateral = np.array([-direction[1], direction[0]])
        ho = np.array([-0.081, -0.593, 0.049])
        hi = ho.copy()
        hi[:2] += direction * 0.010 + lateral * 0.016
        association = associate_targets(
            ho, hi, PASS_GATE, PASS_GATE, alignment_applied=True,
            tool_category="screwdriver", tool_axis_rad=axis)
        self.assertFalse(association["safe_approach_matched"])
        destination, _ = safe_approach_candidate(association)
        self.assertIsNone(destination)

    def test_other_tools_keep_fifteen_millimetre_total_limit(self):
        association = associate_targets(
            [-0.081, -0.593, 0.049], [-0.061, -0.593, 0.049],
            PASS_GATE, PASS_GATE, alignment_applied=True,
            tool_category="tape measure", tool_axis_rad=0.0)
        self.assertFalse(association["safe_approach_matched"])
        destination, _ = safe_approach_candidate(association)
        self.assertIsNone(destination)


if __name__ == "__main__":
    unittest.main()
