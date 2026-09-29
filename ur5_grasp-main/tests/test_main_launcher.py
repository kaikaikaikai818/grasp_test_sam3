import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


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

    def test_interactive_menu_runs_multiple_tools_before_exit(self):
        selections = iter(("1", "卷尺", "q"))
        runner = Mock()
        runner.return_value.returncode = 0

        result = MODULE.interactive_menu(
            input_fn=lambda _prompt: next(selections), runner=runner)

        self.assertEqual(result, 0)
        self.assertEqual(runner.call_count, 2)
        first_command = runner.call_args_list[0].args[0]
        second_command = runner.call_args_list[1].args[0]
        self.assertEqual(first_command[first_command.index("--prompt") + 1], "screwdriver")
        self.assertEqual(second_command[second_command.index("--prompt") + 1], "tape measure")

    def test_invalid_and_pending_choices_return_to_menu(self):
        selections = iter(("unknown", "3", "2", "q"))
        runner = Mock()
        runner.return_value.returncode = 0

        result = MODULE.interactive_menu(
            input_fn=lambda _prompt: next(selections), runner=runner)

        self.assertEqual(result, 0)
        runner.assert_called_once()
        command = runner.call_args.args[0]
        self.assertEqual(command[command.index("--prompt") + 1], "tape measure")

    def test_tool_argument_remains_one_shot(self):
        runner = Mock()
        runner.return_value.returncode = 7
        with patch.object(MODULE, "parse_args",
                          return_value=MODULE.argparse.Namespace(tool="1", list=False)), \
             patch.object(MODULE.subprocess, "run", runner):
            result = MODULE.main()

        self.assertEqual(result, 7)
        runner.assert_called_once()


if __name__ == "__main__":
    unittest.main()
