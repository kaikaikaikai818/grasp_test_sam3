"""统一工具抓取入口，复用已验证的机器人抓取核心。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ROBOT_CORE = PROJECT_ROOT / "ur5_grasp-main" / "grasp_tool.py"


@dataclass(frozen=True)
class ToolEntry:
    key: str
    name: str
    prompt: str
    aliases: tuple[str, ...]
    status: str
    note: str


TOOLS = (
    ToolEntry("1", "螺丝刀", "screwdriver", ("螺丝刀", "screwdriver"),
              "operational", "已通过大、小两把螺丝刀实机回归"),
    ToolEntry("2", "卷尺", "tape measure", ("卷尺", "tape", "tape measure"),
              "operational", "已通过多个角度实机回归"),
    ToolEntry("3", "活动扳手", "adjustable wrench",
              ("活动扳手", "扳手", "adjustable wrench"),
              "pending", "尚未完成视觉、无接触与低力抓取验收"),
    ToolEntry("4", "钳子", "pliers", ("钳子", "pliers"),
              "validation", "先完成视觉、方向与40 mm无接触验收"),
    ToolEntry("5", "胶带切割器", "tape dispenser",
              ("胶带切割器", "胶带座", "tape dispenser"),
              "pending", "尚未完成视觉、无接触与低力抓取验收"),
)


def resolve_tool(value: str) -> ToolEntry | None:
    normalized = value.strip().lower()
    for tool in TOOLS:
        if normalized == tool.key or normalized in {alias.lower() for alias in tool.aliases}:
            return tool
    return None


def build_core_command(tool: ToolEntry, python: str | None = None) -> list[str]:
    if tool.status == "pending":
        raise ValueError(f"{tool.name}尚未通过安全验收，暂不开放机械臂动作。")

    command = [
        python or sys.executable,
        str(ROBOT_CORE),
        "--prompt", tool.prompt,
        "--stage", "grasp",
        "--check-calib",
    ]
    if tool.status == "operational":
        command.append("--auto")
    if tool.prompt == "tape measure":
        command.append("--tape-grasp-test")
    if tool.prompt == "pliers":
        command.append("--pliers-grasp-test")
    return command


def print_menu() -> None:
    print("\n===== 统一工具抓取主程序 =====")
    print("已验证的抓取核心、标定和安全门会被直接复用。\n")
    labels = {"operational": "可运行", "validation": "最终验收", "pending": "尚未开放"}
    for tool in TOOLS:
        print(f"{tool.key}. {tool.name:<8} [{labels[tool.status]}] {tool.note}")
    print("Q. 退出")
    print("提示：工具阶段按 q 会返回本菜单；在本菜单输入 Q 才会完全退出。")


def print_run_instructions(tool: ToolEntry) -> None:
    print(f"\n即将启动：{tool.name}")
    if tool.status == "operational":
        print("等待 D455 目标稳定并确认路径清空后，只需按一次小写 a。")
        print("之后程序自动完成高位观察、D435i 精定位、方向对齐、下降、")
        print("低力夹持和 50 mm 试抬升。完成后按 o 松开，按 q 返回工具菜单。")
    elif tool.prompt == "pliers":
        print("钳子当前处于首次验收：按 P → Y → D，只检查手柄中段和40 mm间隙。")
        print("本次先不要按 R；确认抓取中心位于两条手柄中段之间后再继续。")
    else:
        print("当前工具处于分步验收，先检查视觉、方向和40 mm无接触终点。")
    print("运行前确认示教器活动 TCP 为 TCP_clamp，并保持急停可用。\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一工具抓取主程序")
    parser.add_argument("--tool", help="工具编号或名称；省略时显示选择菜单")
    parser.add_argument("--list", action="store_true", help="只显示工具状态")
    return parser.parse_args()


def run_selected_tool(value: str, runner=None) -> int:
    """Validate one selection and run its isolated grasp process."""
    tool = resolve_tool(value)
    if tool is None:
        print("[错误] 无法识别该工具；请输入菜单编号或工具名称。")
        return 2
    if tool.status == "pending":
        print(f"[未开放] {tool.note}。不会连接机械臂或夹爪。")
        return 3
    if not ROBOT_CORE.is_file():
        print(f"[错误] 找不到抓取核心：{ROBOT_CORE}")
        return 4

    print_run_instructions(tool)
    if runner is None:
        runner = subprocess.run
    return runner(build_core_command(tool), cwd=PROJECT_ROOT).returncode


def interactive_menu(input_fn=None, runner=None) -> int:
    """Keep the launcher alive while each tool runs in a clean subprocess."""
    if input_fn is None:
        input_fn = input

    while True:
        print_menu()
        try:
            value = input_fn("\n请选择工具: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n主程序已退出。")
            return 0

        if value.lower() == "q":
            print("主程序已退出。")
            return 0

        code = run_selected_tool(value, runner=runner)
        if code == 0:
            print("\n[主菜单] 工具阶段已结束，可以继续选择下一件工具。")
        elif code in (2, 3, 4):
            print("[主菜单] 请重新选择工具。")
        else:
            print(f"\n[主菜单] 工具阶段异常结束（代码 {code}），设备已清理；可以重试或退出。")


def main() -> int:
    args = parse_args()
    if args.list:
        print_menu()
        return 0
    if args.tool is not None:
        if args.tool.lower() == "q":
            return 0
        return run_selected_tool(args.tool)
    return interactive_menu()


if __name__ == "__main__":
    raise SystemExit(main())
