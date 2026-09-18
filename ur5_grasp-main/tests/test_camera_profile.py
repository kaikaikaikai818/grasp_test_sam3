from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bsp.camera_bsp.camera_profile import max_intrinsics_delta


class CameraProfileTests(unittest.TestCase):
    def setUp(self):
        self.d435i_calibrated = np.array([
            [610.61681722, 0.0, 332.81388332],
            [0.0, 611.63223595, 248.52879948],
            [0.0, 0.0, 1.0],
        ])

    def test_current_d435i_delta_is_accepted_by_eight_pixel_limit(self):
        live = np.array([
            [604.896, 0.0, 331.726],
            [0.0, 604.523, 249.169],
            [0.0, 0.0, 1.0],
        ])
        delta = max_intrinsics_delta(live, self.d435i_calibrated)
        self.assertGreater(delta, 3.0)
        self.assertLessEqual(delta, 8.0)

    def test_d435i_delta_above_eight_is_rejected(self):
        live = self.d435i_calibrated.copy()
        live[0, 0] += 8.01
        self.assertGreater(max_intrinsics_delta(live, self.d435i_calibrated), 8.0)

    def test_d455_delta_above_three_is_rejected(self):
        expected = np.array([[386.471, 0, 321.617], [0, 386.034, 237.200], [0, 0, 1]])
        live = expected.copy()
        live[1, 1] += 3.01
        self.assertGreater(max_intrinsics_delta(live, expected), 3.0)

    def test_invalid_intrinsics_are_rejected(self):
        live = self.d435i_calibrated.copy()
        live[0, 0] = np.nan
        self.assertEqual(max_intrinsics_delta(live, self.d435i_calibrated), float("inf"))


if __name__ == "__main__":
    unittest.main()
