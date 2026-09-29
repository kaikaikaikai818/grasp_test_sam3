"""Guarded fixed-zone handover with directional pull-to-release detection."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def _finite_vector(value, size, label):
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != size or not np.all(np.isfinite(vector)):
        raise ValueError("invalid %s" % label)
    return vector


def load_handover_config(path: Path, category: str, workspace_limits):
    """Load an explicitly approved cell and tool-specific handover pose."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("approved_for_this_cell") is not True:
        raise ValueError("handover is not approved for this cell")
    tool = data.get("tools", {}).get(category)
    if not isinstance(tool, dict) or tool.get("approved") is not True:
        raise ValueError("handover pose is not approved for tool: " + category)

    xyz = _finite_vector(data.get("handover_xyz_m"), 3, "handover xyz")
    orientation = _finite_vector(tool.get("orientation_rvec"), 3, "handover orientation")
    direction = _finite_vector(data.get("pull_direction_base_xy"), 2, "pull direction")
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 0.9:
        raise ValueError("pull direction must be a non-zero base XY vector")
    direction /= direction_norm

    limits = np.asarray(workspace_limits, dtype=np.float64)
    if any(not limits[i, 0] <= xyz[i] <= limits[i, 1] for i in range(3)):
        raise ValueError("handover point outside workspace")
    travel_z = float(data["travel_tcp_z_m"])
    if not np.isfinite(travel_z) or travel_z < xyz[2] + 0.10 or travel_z > limits[2, 1]:
        raise ValueError("handover travel height has insufficient clearance")

    config = {
        "category": category,
        "handover_pose": xyz.tolist() + orientation.tolist(),
        "travel_tcp_z_m": travel_z,
        "pull_direction_base_xy": direction.tolist(),
        "pull_threshold_n": float(data["pull_threshold_n"]),
        "max_lateral_force_n": float(data["max_lateral_force_n"]),
        "pull_sustain_s": float(data["pull_sustain_s"]),
        "baseline_duration_s": float(data["baseline_duration_s"]),
        "handover_timeout_s": float(data["handover_timeout_s"]),
        "retreat_distance_m": float(data.get("retreat_distance_m", 0.08)),
    }
    calibration = data.get("force_calibration", {})
    light_peaks = _finite_vector(
        calibration.get("light_touch_peak_n"), 3, "three light-touch trials")
    take_peaks = _finite_vector(
        calibration.get("normal_take_peak_n"), 3, "three normal-take trials")
    take_durations = _finite_vector(
        calibration.get("normal_take_duration_s"), 3,
        "three normal-take durations")
    if not 3.0 <= config["pull_threshold_n"] <= 30.0:
        raise ValueError("pull threshold outside calibrated range")
    if not 3.0 <= config["max_lateral_force_n"] <= 30.0:
        raise ValueError("lateral force limit outside calibrated range")
    if not 0.10 <= config["pull_sustain_s"] <= 1.0:
        raise ValueError("pull sustain time outside safe range")
    if not 0.5 <= config["baseline_duration_s"] <= 3.0:
        raise ValueError("force baseline duration outside safe range")
    if not 5.0 <= config["handover_timeout_s"] <= 60.0:
        raise ValueError("handover timeout outside safe range")
    if not 0.04 <= config["retreat_distance_m"] <= 0.15:
        raise ValueError("handover retreat distance outside safe range")
    if (np.any(light_peaks < 0) or np.any(take_peaks <= 0)
            or np.any(take_durations <= 0)):
        raise ValueError("handover force calibration must contain positive trials")
    if not (float(np.max(light_peaks)) < config["pull_threshold_n"]
            <= float(np.min(take_peaks))):
        raise ValueError("pull threshold does not separate light touch and normal take")
    if config["pull_sustain_s"] > float(np.min(take_durations)):
        raise ValueError("pull sustain time exceeds a calibrated normal take")
    return config


class DirectionalPullDetector:
    """Require a sustained pull toward the taught person direction."""

    def __init__(self, baseline_wrench, direction_xy, threshold_n,
                 max_lateral_force_n, sustain_s):
        self.baseline = _finite_vector(baseline_wrench, 6, "baseline wrench")
        direction = _finite_vector(direction_xy, 2, "pull direction")
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-9:
            raise ValueError("pull direction must be non-zero")
        self.direction = direction / direction_norm
        self.threshold_n = float(threshold_n)
        self.max_lateral_force_n = float(max_lateral_force_n)
        self.sustain_s = float(sustain_s)
        self.started_at = None

    def update(self, wrench, now):
        delta_xy = _finite_vector(wrench, 6, "TCP wrench")[:2] - self.baseline[:2]
        pull_n = float(np.dot(delta_xy, self.direction))
        lateral_n = float(np.linalg.norm(delta_xy - pull_n * self.direction))
        valid = pull_n >= self.threshold_n and lateral_n <= self.max_lateral_force_n
        if not valid:
            self.started_at = None
            return False, pull_n, lateral_n
        if self.started_at is None:
            self.started_at = float(now)
        return float(now) - self.started_at >= self.sustain_s, pull_n, lateral_n


def collect_force_baseline(robot, duration_s, sample_interval_s=0.02,
                           monotonic=time.monotonic, sleep=time.sleep):
    deadline = monotonic() + float(duration_s)
    samples = []
    while monotonic() < deadline:
        samples.append(_finite_vector(robot.get_actual_tcp_force(), 6, "TCP wrench"))
        sleep(sample_interval_s)
    if len(samples) < 5:
        raise RuntimeError("not enough force samples for handover baseline")
    return np.median(np.asarray(samples), axis=0)


def wait_for_directional_pull(robot, config, sample_interval_s=0.02,
                              monotonic=time.monotonic, sleep=time.sleep):
    """Wait while stationary; return (confirmed, reason) without moving or releasing."""
    try:
        baseline = collect_force_baseline(
            robot, config["baseline_duration_s"], sample_interval_s,
            monotonic=monotonic, sleep=sleep)
        detector = DirectionalPullDetector(
            baseline, config["pull_direction_base_xy"], config["pull_threshold_n"],
            config["max_lateral_force_n"], config["pull_sustain_s"])
        deadline = monotonic() + config["handover_timeout_s"]
        while monotonic() < deadline:
            confirmed, _, _ = detector.update(robot.get_actual_tcp_force(), monotonic())
            if confirmed:
                return True, None
            sleep(sample_interval_s)
        return False, "handover pull timed out; gripper remains closed"
    except Exception as exc:
        return False, "force feedback unavailable: %s" % exc


def execute_fixed_handover(robot, config, workspace_limits, open_position,
                           open_speed=30, open_force=20,
                           wait_for_pull=wait_for_directional_pull):
    """Move to a taught handover pose, release only after pull, then retreat."""
    current = _finite_vector(robot.get_actual_tcp_pose(), 6, "current TCP pose")
    limits = np.asarray(workspace_limits, dtype=np.float64)
    pose = np.asarray(config["handover_pose"], dtype=np.float64)
    travel_z = max(float(current[2]), float(config["travel_tcp_z_m"]))
    high_current = current.copy()
    high_current[2] = travel_z
    high_handover = pose.copy()
    high_handover[2] = travel_z
    direction = np.asarray(config["pull_direction_base_xy"], dtype=np.float64)
    retreat = pose.copy()
    retreat[:2] -= direction * config["retreat_distance_m"]
    planned_points = (high_current[:3], high_handover[:3], pose[:3], retreat[:3])
    if any(any(not limits[i, 0] <= point[i] <= limits[i, 1] for i in range(3))
           for point in planned_points):
        raise ValueError("handover path outside workspace")

    robot.moveL(high_current.tolist(), speed=0.10, acceleration=0.10)
    robot.moveL(high_handover.tolist(), speed=0.10, acceleration=0.10)
    robot.moveL(pose.tolist(), speed=0.03, acceleration=0.03)
    confirmed, reason = wait_for_pull(robot, config)
    if not confirmed:
        return False, reason
    if robot.grip(open_position, open_speed, open_force) == -1:
        return False, "gripper rejected handover release command"

    robot.moveL(retreat.tolist(), speed=0.03, acceleration=0.03)
    retreat[2] = travel_z
    robot.moveL(retreat.tolist(), speed=0.10, acceleration=0.10)
    return True, None
