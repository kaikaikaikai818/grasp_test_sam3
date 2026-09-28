from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp_tool import grip_contact_confirmed


class GripContactTests(unittest.TestCase):
    def test_controller_contact_flag_is_sufficient(self):
        confirmed, shortfall = grip_contact_confirmed(1, 0, 11000, 11000, 30)
        self.assertTrue(confirmed)
        self.assertEqual(shortfall, 0)

    def test_low_force_screwdriver_contact_uses_torque_and_blockage(self):
        confirmed, shortfall = grip_contact_confirmed(0, 38, 9500, 11000, 30)
        self.assertTrue(confirmed)
        self.assertEqual(shortfall, 1500)

    def test_empty_full_close_is_not_contact(self):
        confirmed, _ = grip_contact_confirmed(0, 38, 11000, 11000, 30)
        self.assertFalse(confirmed)

    def test_blockage_without_minimum_torque_is_not_contact(self):
        confirmed, _ = grip_contact_confirmed(0, 29, 9500, 11000, 30)
        self.assertFalse(confirmed)


if __name__ == "__main__":
    unittest.main()
