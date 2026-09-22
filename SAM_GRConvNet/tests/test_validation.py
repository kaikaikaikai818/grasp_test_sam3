import unittest
import numpy as np

from validate_folder import axial_error_deg


class ValidationTests(unittest.TestCase):
    def test_axial_angle_wraps_at_180_degrees(self):
        self.assertAlmostEqual(axial_error_deg(np.radians(89), np.radians(-91)), 0)
        self.assertAlmostEqual(axial_error_deg(np.radians(10), np.radians(20)), 10)


if __name__ == "__main__":
    unittest.main()
