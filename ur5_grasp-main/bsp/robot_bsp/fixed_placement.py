"""Guarded fixed-location release after a confirmed grasp.

Coordinates must be taught and checked on the physical cell. No built-in
destination is provided because an invented pose could strike the table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def plan_safe_orientation_return(current_pose, target_orientation, safe_z, workspace_limits):
    """Create vertical-lift and in-place-turn poses without lowering the TCP."""
    current = np.asarray(current_pose, dtype=float).reshape(-1)
    orientation = np.asarray(target_orientation, dtype=float).reshape(-1)
    if current.size != 6 or orientation.size != 3 or not np.all(np.isfinite(current)) \
            or not np.all(np.isfinite(orientation)):
        raise ValueError("invalid current or target orientation")
    limits = np.asarray(workspace_limits, dtype=float)
    lift_z = max(float(current[2]), float(safe_z))
    if any(not (limits[i, 0] <= value <= limits[i, 1])
           for i, value in enumerate((current[0], current[1], lift_z))):
        raise ValueError("orientation return pose outside workspace")
    lift = current.copy()
    lift[2] = lift_z
    turn = lift.copy()
    turn[3:6] = orientation
    return lift.tolist(), turn.tolist()


def load_placement(path: Path, workspace_limits):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("approved_for_this_cell") is not True:
        raise ValueError("placement is not approved for this cell")
    try:
        values = np.asarray([data[k] for k in ("x_m", "y_m", "release_tcp_z_m",
                                               "travel_tcp_z_m", "surface_z_m")], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("placement needs taught numeric coordinates") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("placement has non-finite coordinates")
    x, y, release_z, travel_z, surface_z = values
    if release_z < surface_z + 0.005 or travel_z < release_z + 0.10:
        raise ValueError("placement has insufficient vertical clearance")
    limits = np.asarray(workspace_limits, dtype=float)
    if any(not (limits[i, 0] <= value <= limits[i, 1])
           for i, value in enumerate((x, y, release_z))):
        raise ValueError("placement release point outside workspace")
    if travel_z > limits[2, 1]:
        raise ValueError("placement travel height outside workspace")
    return dict(zip(("x_m", "y_m", "release_tcp_z_m", "travel_tcp_z_m",
                     "surface_z_m"), values.tolist()))


def place_at_fixed_point(robot, placement, workspace_limits, open_position,
                         open_speed, open_force, speed=0.02):
    current = np.asarray(robot.get_actual_tcp_pose(), dtype=float)
    if current.shape != (6,) or not np.all(np.isfinite(current)):
        raise ValueError("invalid current TCP pose")
    x, y = placement["x_m"], placement["y_m"]
    travel_z = max(float(current[2]), placement["travel_tcp_z_m"])
    limits = np.asarray(workspace_limits, dtype=float)
    if travel_z > limits[2, 1]:
        raise ValueError("travel height outside workspace")
    if any(not (limits[i, 0] <= current[i] <= limits[i, 1]) for i in range(3)):
        raise ValueError("current TCP outside workspace")
    orientation = current[3:6].tolist()
    robot.moveL([current[0], current[1], travel_z] + orientation,
                speed=speed, acceleration=speed)
    robot.moveL([x, y, travel_z] + orientation, speed=speed, acceleration=speed)
    robot.moveL([x, y, placement["release_tcp_z_m"]] + orientation,
                speed=speed, acceleration=speed)
    result = robot.grip(open_position, open_speed, open_force)
    if result == -1:
        raise RuntimeError("gripper did not accept release command")
    robot.moveL([x, y, travel_z] + orientation, speed=speed, acceleration=speed)
