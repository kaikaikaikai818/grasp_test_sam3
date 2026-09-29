from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp_tool import grip_contact_confirmed, wait_for_grip_contact


class GripContactTests(unittest.TestCase):
    def test_controller_contact_flag_is_sufficient(self):
        self.assertTrue(grip_contact_confirmed(1, 0, 30))

    def test_low_force_screwdriver_contact_uses_torque(self):
        self.assertTrue(grip_contact_confirmed(0, 38, 30))

    def test_below_minimum_torque_is_not_contact(self):
        self.assertFalse(grip_contact_confirmed(0, 29, 30))

    def test_strict_contact_rejects_flag_without_force(self):
        self.assertFalse(grip_contact_confirmed(1, 20, 80, mode="both"))

    def test_strict_contact_rejects_mechanical_stop_at_threshold(self):
        self.assertFalse(grip_contact_confirmed(1, 80, 80, mode="both"))

    def test_strict_contact_rejects_force_without_flag(self):
        self.assertFalse(grip_contact_confirmed(0, 120, 80, mode="both"))

    def test_strict_contact_requires_flag_and_force(self):
        self.assertTrue(grip_contact_confirmed(1, 120, 80, mode="both"))

    def test_contact_wait_allows_slow_fingers_to_reach_object(self):
        class Robot:
            reached = iter((0, 0, 1))
            current = iter((47, 70, 105))

            def read_torque_reached(self):
                return next(self.reached)

            def read_torque_current(self):
                return next(self.current)

        confirmed, reached, current = wait_for_grip_contact(
            Robot(), 80, mode="both", max_samples=3, poll_interval_s=0)
        self.assertTrue(confirmed)
        self.assertEqual((reached, current), (1, 105))

    def test_contact_wait_rejects_persistent_empty_close(self):
        class Robot:
            def read_torque_reached(self):
                return 0

            def read_torque_current(self):
                return 47

        confirmed, reached, current = wait_for_grip_contact(
            Robot(), 80, mode="both", max_samples=3, poll_interval_s=0)
        self.assertFalse(confirmed)
        self.assertEqual((reached, current), (0, 47))


if __name__ == "__main__":
    unittest.main()
