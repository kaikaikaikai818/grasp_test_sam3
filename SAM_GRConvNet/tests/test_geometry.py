import unittest

import numpy as np
import cv2

from grasp_backend.geometry import GeometryGraspPredictor, geometry_maps


class GeometryGraspTests(unittest.TestCase):
    def test_long_tool_grasp_is_on_narrow_handle(self):
        mask = np.zeros((160, 100), dtype=bool)
        mask[15:55, 25:75] = True
        mask[45:145, 43:57] = True
        maps = geometry_maps(mask)
        row, column = np.unravel_index(np.argmax(maps["quality"]), mask.shape)

        self.assertGreater(row, 55)
        self.assertTrue(mask[row, column])
        self.assertLess(maps["width"][row, column], 25)
        self.assertLess(abs(np.degrees(maps["angle"][row, column])), 10)

    def test_predictor_returns_safe_candidate(self):
        image = np.zeros((100, 160, 3), dtype=np.uint8)
        mask = np.zeros((100, 160), dtype=bool)
        mask[42:58, 15:145] = True

        result = GeometryGraspPredictor().predict(image, "tool", [mask])

        self.assertTrue(result.grasps[0].accepted)
        self.assertEqual(result.backend, "mask geometry baseline")
        self.assertTrue(mask[result.grasps[0].candidate.center_xy[1],
                             result.grasps[0].candidate.center_xy[0]])

    def test_diagonal_wrench_avoids_large_head(self):
        mask = np.zeros((300, 220), dtype=np.uint8)
        cv2.rectangle(mask, (95, 65), (125, 265), 1, -1)
        cv2.rectangle(mask, (55, 25), (165, 105), 1, -1)
        matrix = cv2.getRotationMatrix2D((110, 150), -18, 1.0)
        mask = cv2.warpAffine(mask, matrix, (220, 300), flags=cv2.INTER_NEAREST).astype(bool)

        maps = geometry_maps(mask)
        row, column = np.unravel_index(np.argmax(maps["quality"]), mask.shape)

        self.assertGreater(row, 115)
        self.assertLess(maps["width"][row, column], 55)


if __name__ == "__main__":
    unittest.main()
