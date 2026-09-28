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
              "validation", "已能抓取，仍需完成两次最终回归"),
    ToolEntry("3", "活动扳手", "adjustable wrench",
              ("活动扳手", "扳手", "adjustable wrench"),
              "pending", "尚未完成视觉、无接触与低力抓取验收"),
    ToolEntry("4", "钳子", "pliers", ("钳子", "pliers"),
              "pending", "尚未完成视觉、无接触与低力抓取验收"),
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
    return command


def print_menu() -> None:
    print("\n===== 统一工具抓取主程序 =====")
    print("已验证的抓取核心、标定和安全门会被直接复用。\n")
    labels = {"operational": "可运行", "validation": "最终验收", "pending": "尚未开放"}
    for tool in TOOLS:
        print(f"{tool.key}. {tool.name:<8} [{labels[tool.status]}] {tool.note}")
    print("Q. 退出")


def print_run_instructions(tool: ToolEntry) -> None:
    print(f"\n即将启动：{tool.name}")
    if tool.status == "operational":
        print("等待 D455 目标稳定并确认路径清空后，只需按一次小写 a。")
        print("之后程序自动完成高位观察、D435i 精定位、方向对齐、下降、")
        print("低力夹持和 50 mm 试抬升。完成后按 o 松开，按 q 退出。")
    else:
        print("卷尺当前仍处于最终验收：按 P → Y → D，检查 40 mm 间隙后按 R。")
        print("成功后按 O 松开、按 U 回升；连续两次成功后再切换为自动流程。")
    print("运行前确认示教器活动 TCP 为 TCP_clamp，并保持急停可用。\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一工具抓取主程序")
    parser.add_argument("--tool", help="工具编号或名称；省略时显示选择菜单")
    parser.add_argument("--list", action="store_true", help="只显示工具状态")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        print_menu()
        return 0

    value = args.tool
    if value is None:
        print_menu()
        value = input("\n请选择工具: ").strip()
    if value.lower() == "q":
        return 0

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
    return subprocess.run(build_core_command(tool), cwd=PROJECT_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
