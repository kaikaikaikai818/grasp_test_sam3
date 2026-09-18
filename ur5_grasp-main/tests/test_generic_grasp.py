"""Hardware-free checks for the conservative generic grasp geometry."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bsp.camera_bsp.generic_grasp import (base_grasp_plan, pixel_grasp_candidate,
                                          select_one)


class GenericGraspTests(unittest.TestCase):
    def elongated_result(self):
        mask = np.zeros((120, 160), dtype=bool)
        mask[48:70, 25:135] = True
        return {"mask": mask, "score": 0.8, "z_mm": 500.0}

    def test_single_elongated_mask_produces_central_candidate(self):
        result = self.elongated_result()
        depth = np.full(result["mask"].shape, 500, dtype=np.uint16)
        candidate, reason = pixel_grasp_candidate(result, depth, 0.001)
        self.assertIsNotNone(candidate, reason)
        self.assertLess(abs(candidate.center_px[0] - 80), 3)
        plan, reason = base_grasp_plan(
            candidate, np.array([[600, 0, 80], [0, 600, 60], [0, 0, 1]]),
            lambda u, v, z: np.array([(u - 80) * z / 600, (v - 60) * z / 600, z]),
            0.005, 0.050)
        self.assertIsNotNone(plan, reason)

    def test_round_mask_is_rejected(self):
        yy, xx = np.ogrid[:100, :100]
        mask = (xx - 50) ** 2 + (yy - 50) ** 2 < 25 ** 2
        candidate, reason = pixel_grasp_candidate(
            {"mask": mask}, np.full(mask.shape, 500, dtype=np.uint16), 0.001)
        self.assertIsNone(candidate)
        self.assertIn("elongated", reason)

    def test_multiple_instances_are_never_selected(self):
        result, reason = select_one(
            [{"score": 0.8, "z_mm": 500}, {"score": 0.7, "z_mm": 510}], 0.4)
        self.assertIsNone(result)
        self.assertIn("multiple", reason)


if __name__ == "__main__":
    unittest.main()
