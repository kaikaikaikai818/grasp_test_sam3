import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bsp.robot_bsp.fixed_placement import (load_placement, place_at_fixed_point,
                                           plan_safe_orientation_return)


LIMITS = [[-0.5, 0.05], [-0.8, -0.45], [-0.2, 0.6]]


class FakeRobot:
    def __init__(self):
        self.moves = []
        self.released = False

    def get_actual_tcp_pose(self):
        return [-0.3, -0.6, 0.12, 3.141, 0, 0]

    def moveL(self, pose, **kwargs):
        self.moves.append(pose)

    def grip(self, *args):
        self.released = True
        return 6000


class PlacementTests(unittest.TestCase):
    def test_orientation_return_lifts_before_turning(self):
        current = [-.3, -.6, .08, 2.0, .1, .2]
        lift, turn = plan_safe_orientation_return(current, [3.141, 0, 0], .14, LIMITS)
        self.assertEqual(lift[:2], current[:2])
        self.assertEqual(turn[:3], lift[:3])
        self.assertEqual(lift[3:], current[3:])
        self.assertEqual(turn[3:], [3.141, 0, 0])

    def test_unapproved_or_unset_destination_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "place.json"
            path.write_text(json.dumps({"approved_for_this_cell": False}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_placement(path, LIMITS)

    def test_vertical_then_horizontal_then_release_then_retreat(self):
        robot = FakeRobot()
        placement = {"x_m": -.2, "y_m": -.6, "release_tcp_z_m": .02,
                     "travel_tcp_z_m": .3, "surface_z_m": 0.0}
        place_at_fixed_point(robot, placement, LIMITS, 6000, 100, 40)
        self.assertEqual(len(robot.moves), 4)
        self.assertEqual(robot.moves[0][:2], [-.3, -.6])
        self.assertEqual(robot.moves[1][:3], [-.2, -.6, .3])
        self.assertTrue(robot.released)
        self.assertEqual(robot.moves[-1][:3], [-.2, -.6, .3])


if __name__ == "__main__":
    unittest.main()
