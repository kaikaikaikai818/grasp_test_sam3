import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bsp.camera_bsp.tool_profiles import load_tool_profile

LIMITS = [[-0.5, 0.05], [-0.8, -0.45], [-0.2, 0.6]]


class ProfileTests(unittest.TestCase):
    def test_unapproved_profile_cannot_enable_grasp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            path.write_text(json.dumps({"tools": {"adjustable wrench": {
                "approved_for_this_cell": False}}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_tool_profile(path, "adjustable wrench", LIMITS)

    def test_approved_profile_validates_range(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            data = {"approved_for_this_cell": True, "grasp_tcp_z_m": 0.03,
                    "open_position": 6000, "close_position": 11000,
                    "grip_force": 30, "requires_angle": True}
            path.write_text(json.dumps({"tools": {"adjustable wrench": data}}),
                            encoding="utf-8")
            profile = load_tool_profile(path, "adjustable wrench", LIMITS)
            self.assertTrue(profile["requires_angle"])
            self.assertEqual(profile["close_position"], 11000)


if __name__ == "__main__":
    unittest.main()
