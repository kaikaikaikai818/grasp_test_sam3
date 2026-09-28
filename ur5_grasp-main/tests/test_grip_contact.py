from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp_tool import grip_contact_confirmed


class GripContactTests(unittest.TestCase):
    def test_controller_contact_flag_is_sufficient(self):
        self.assertTrue(grip_contact_confirmed(1, 0, 30))

    def test_low_force_screwdriver_contact_uses_torque(self):
        self.assertTrue(grip_contact_confirmed(0, 38, 30))

    def test_below_minimum_torque_is_not_contact(self):
        self.assertFalse(grip_contact_confirmed(0, 29, 30))


if __name__ == "__main__":
    unittest.main()
