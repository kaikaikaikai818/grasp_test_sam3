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


if __name__ == "__main__":
    unittest.main()
