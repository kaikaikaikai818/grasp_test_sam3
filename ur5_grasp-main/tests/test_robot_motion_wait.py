import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bsp.robot_bsp.UR_Robot import UR_Robot


class MotionWaitTests(unittest.TestCase):
    def test_move_l_requires_consecutive_stable_samples_without_fixed_dwell(self):
        robot = UR_Robot.__new__(UR_Robot)
        robot.tool_pose_tolerance = [0.002] * 3 + [0.01] * 3
        target = [0.1, -0.6, 0.25, 3.141, 0.0, 0.0]
        samples = [
            [0.08, -0.6, 0.25, 3.141, 0.0, 0.0],
            target,
            target,
            target,
        ]

        class Control:
            def __init__(self):
                self.calls = []

            def moveL(self, pose, speed, acceleration):
                self.calls.append((pose, speed, acceleration))

        robot.rtde_c = Control()
        robot.get_actual_tcp_pose = lambda: samples.pop(0)
        robot.moveL(target, speed=0.2, acceleration=0.2)
        self.assertEqual(len(robot.rtde_c.calls), 1)
        self.assertEqual(samples, [])

    def test_invalid_force_vector_is_rejected(self):
        robot = UR_Robot.__new__(UR_Robot)

        class Receive:
            def getActualTCPForce(self):
                return [1.0, 2.0]

        robot.rtde_r = Receive()
        with self.assertRaisesRegex(RuntimeError, "invalid TCP force"):
            robot.get_actual_tcp_force()

    def test_equivalent_rotation_vector_does_not_add_fixed_wait(self):
        robot = UR_Robot.__new__(UR_Robot)
        robot.tool_pose_tolerance = [0.002] * 3 + [0.01] * 3
        target = [0.1, -0.6, 0.25, 3.1415926, 0.0, 0.0]
        equivalent = [0.1, -0.6, 0.25, -3.1415927, 0.0, 0.0]

        class Control:
            def moveL(self, pose, speed, acceleration):
                pass

        robot.rtde_c = Control()
        robot.get_actual_tcp_pose = lambda: equivalent
        robot.moveL(target)


if __name__ == "__main__":
    unittest.main()
