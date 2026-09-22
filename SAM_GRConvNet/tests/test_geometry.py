import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
