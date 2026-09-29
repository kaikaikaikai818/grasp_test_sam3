import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("main_launcher", ROOT / "run.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MainLauncherTests(unittest.TestCase):
    def test_resolves_chinese_name_and_number(self):
        self.assertEqual(MODULE.resolve_tool("螺丝刀").prompt, "screwdriver")
        self.assertEqual(MODULE.resolve_tool("2").prompt, "tape measure")
        self.assertEqual(MODULE.resolve_tool("钳子").prompt, "pliers")

    def test_screwdriver_uses_automatic_validated_flow(self):
        command = MODULE.build_core_command(MODULE.resolve_tool("1"), python="python")
        self.assertIn("--auto", command)
        self.assertNotIn("--tape-grasp-test", command)
        self.assertEqual(command[command.index("--prompt") + 1], "screwdriver")

    def test_tape_uses_automatic_validated_flow(self):
        command = MODULE.build_core_command(MODULE.resolve_tool("卷尺"), python="python")
        self.assertIn("--auto", command)
        self.assertIn("--tape-grasp-test", command)

    def test_pliers_uses_manual_validation_profile(self):
        command = MODULE.build_core_command(MODULE.resolve_tool("钳子"), python="python")
        self.assertNotIn("--auto", command)
        self.assertIn("--pliers-grasp-test", command)

    def test_unvalidated_tool_never_builds_robot_command(self):
        with self.assertRaisesRegex(ValueError, "暂不开放"):
            MODULE.build_core_command(MODULE.resolve_tool("活动扳手"), python="python")


if __name__ == "__main__":
    unittest.main()
