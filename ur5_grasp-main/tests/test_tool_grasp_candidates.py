import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bsp.camera_bsp.tool_grasp_candidates import propose_grasp_region


class CandidateTests(unittest.TestCase):
    def test_wrench_candidate_avoids_wide_head(self):
        mask = np.zeros((120, 160), dtype=bool)
        mask[40:80, 12:42] = True
        mask[55:65, 40:145] = True
        candidate, reason = propose_grasp_region(
            mask, np.full(mask.shape, 1000, np.uint16), .001,
            "adjustable wrench")
        self.assertIsNone(reason)
        self.assertGreater(candidate.center_px[0], 65)

    def test_ambiguous_shape_is_rejected(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        candidate, reason = propose_grasp_region(
            mask, np.full(mask.shape, 1000, np.uint16), .001,
            "rubber mallet")
        self.assertIsNone(candidate)
        self.assertIn("direction", reason)

    def test_tape_measure_uses_deepest_interior_body_region(self):
        mask = np.zeros((120, 160), dtype=bool)
        mask[25:95, 35:125] = True
        mask[55:65, 125:150] = True  # protruding tape outlet
        candidate, reason = propose_grasp_region(
            mask, np.full(mask.shape, 600, np.uint16), .001,
            "tape measure")
        self.assertIsNone(reason)
        self.assertLess(candidate.center_px[0], 125)
        self.assertGreater(candidate.region_radius_px, 20)

    def test_tape_measure_rejects_thin_or_fragmented_body(self):
        mask = np.zeros((80, 120), dtype=bool)
        mask[38:42, 10:110] = True
        candidate, reason = propose_grasp_region(
            mask, np.full(mask.shape, 600, np.uint16), .001,
            "tape measure")
        self.assertIsNone(candidate)
        self.assertIn("body area", reason)


if __name__ == "__main__":
    unittest.main()
