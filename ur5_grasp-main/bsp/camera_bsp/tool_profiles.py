"""Cell-specific gripping parameters for non-screwdriver tabletop tools."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .tool_names import PROMPTS


def load_tool_profile(path: Path, category: str, workspace_limits):
    if category not in PROMPTS or category == "screwdriver":
        raise ValueError("a separate approved profile is required for this category")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    profile = data.get("tools", {}).get(category)
    if not isinstance(profile, dict) or profile.get("approved_for_this_cell") is not True:
        raise ValueError("tool profile has not been approved for this cell: " + category)
    fields = ("grasp_tcp_z_m", "open_position", "close_position", "grip_force")
    try:
        values = [float(profile[field]) for field in fields]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("tool profile has missing numeric grip parameters") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("tool profile has non-finite grip parameters")
    z, open_position, close_position, force = values
    if not workspace_limits[2][0] <= z <= workspace_limits[2][1]:
        raise ValueError("tool grip TCP height outside workspace")
    if not 0 <= open_position <= 65535 or not 0 <= close_position <= 65535:
        raise ValueError("gripper position outside device range")
    if not 1 <= force <= 100:
        raise ValueError("grip force outside 1–100 range")
    torque_min = int(profile.get("torque_min", 80))
    if not 0 <= torque_min <= 65535:
        raise ValueError("grip torque threshold outside device range")
    return {
        "grasp_tcp_z_m": z, "open_position": int(open_position),
        "close_position": int(close_position), "grip_force": int(force),
        "torque_min": torque_min,
        "requires_angle": bool(profile.get("requires_angle", category != "tape measure")),
    }
