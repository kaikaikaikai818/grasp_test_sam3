"""Validated, tool-independent alignment between D455 and D435i base estimates."""

import hashlib
import json
from pathlib import Path

import numpy as np


ALIGNMENT_VERSION = 2
VALIDATION_MEDIAN_LIMIT_M = 0.010
VALIDATION_MAX_LIMIT_M = 0.015


def fit_planar_alignment(d455_xyz, d435i_xyz):
    """Fit XY rigid transform plus robust Z offset, mapping D455 to D435i."""
    source = np.asarray(d455_xyz, dtype=np.float64)
    target = np.asarray(d435i_xyz, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("校正坐标必须是形状相同的 N×3 数组。")
    if source.shape[0] < 3:
        raise ValueError("至少需要3个不同位置才能计算双相机相对校正。")

    source_xy = source[:, :2]
    target_xy = target[:, :2]
    source_center = source_xy.mean(axis=0)
    target_center = target_xy.mean(axis=0)
    source_zero = source_xy - source_center
    target_zero = target_xy - target_center
    if np.linalg.matrix_rank(source_zero) < 2:
        raise ValueError("校正位置分布接近一条直线，请使用中心、左右、上下位置。")

    u, _, vt = np.linalg.svd(source_zero.T @ target_zero)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    z_offset = float(np.median(target[:, 2] - source[:, 2]))

    aligned = apply_alignment_array(source, rotation, translation, z_offset)
    residuals = np.linalg.norm(aligned - target, axis=1)
    return {
        "version": ALIGNMENT_VERSION,
        "mode": "fitted_correction",
        "rotation_xy": rotation.tolist(),
        "translation_xy_m": translation.tolist(),
        "z_offset_m": z_offset,
        "fit_residual_m": residuals.tolist(),
        "fit_residual_median_m": float(np.median(residuals)),
        "fit_residual_max_m": float(np.max(residuals)),
    }


def identity_alignment(d455_xyz, d435i_xyz):
    """Keep existing hand-eye transforms when their raw agreement is sufficient."""
    metrics = alignment_metrics(d455_xyz, d435i_xyz)
    return {
        "version": ALIGNMENT_VERSION,
        "mode": "validated_identity",
        "rotation_xy": [[1.0, 0.0], [0.0, 1.0]],
        "translation_xy_m": [0.0, 0.0],
        "z_offset_m": 0.0,
        "fit_residual_m": metrics["residual_m"],
        "fit_residual_median_m": metrics["median_m"],
        "fit_residual_max_m": metrics["max_m"],
    }


def select_candidate_alignment(d455_xyz, d435i_xyz):
    """Select identity when raw calibration passes, otherwise fit one correction."""
    raw = alignment_metrics(d455_xyz, d435i_xyz)
    alignment = (
        identity_alignment(d455_xyz, d435i_xyz)
        if raw["max_m"] <= VALIDATION_MAX_LIMIT_M
        else fit_planar_alignment(d455_xyz, d435i_xyz)
    )
    alignment["raw_residual_m"] = raw["residual_m"]
    alignment["raw_residual_median_m"] = raw["median_m"]
    alignment["raw_residual_max_m"] = raw["max_m"]
    return alignment


def apply_alignment_array(points, rotation, translation, z_offset):
    points = np.asarray(points, dtype=np.float64)
    output = points.copy()
    output[:, :2] = (np.asarray(rotation) @ points[:, :2].T).T + translation
    output[:, 2] += float(z_offset)
    return output


def apply_alignment(point_xyz, alignment):
    point = np.asarray(point_xyz, dtype=np.float64).reshape(1, 3)
    aligned = apply_alignment_array(
        point, alignment["rotation_xy"], alignment["translation_xy_m"],
        alignment["z_offset_m"])
    return tuple(float(value) for value in aligned[0])


def alignment_metrics(d455_xyz, d435i_xyz, alignment=None):
    source = np.asarray(d455_xyz, dtype=np.float64)
    target = np.asarray(d435i_xyz, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("验证坐标必须是形状相同的 N×3 数组。")
    compared = source
    if alignment is not None:
        compared = apply_alignment_array(
            source, alignment["rotation_xy"], alignment["translation_xy_m"],
            alignment["z_offset_m"])
    residuals = np.linalg.norm(compared - target, axis=1)
    return {
        "residual_m": [float(value) for value in residuals],
        "median_m": float(np.median(residuals)),
        "max_m": float(np.max(residuals)),
    }


def validation_passes(metrics):
    try:
        median_m = float(metrics["median_m"])
        max_m = float(metrics["max_m"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        np.isfinite(median_m) and np.isfinite(max_m)
        and median_m <= VALIDATION_MEDIAN_LIMIT_M
        and max_m <= VALIDATION_MAX_LIMIT_M)


def file_sha256(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("核心标定文件不存在：%s" % path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_calibration_context(d455_serial, d435i_serial, d455_size, d435i_size,
                              camera_pose_path, cam2end_path, camera_ini_path):
    """Describe the exact hardware/calibration state authorized by validation."""
    return {
        "d455_serial": str(d455_serial),
        "d435i_serial": str(d435i_serial),
        "d455_size": [int(d455_size[0]), int(d455_size[1])],
        "d435i_size": [int(d435i_size[0]), int(d435i_size[1])],
        "core_calibration_sha256": {
            "camera_pose.txt": file_sha256(camera_pose_path),
            "cam2end_20260906.txt": file_sha256(cam2end_path),
            "camera_20260906.ini": file_sha256(camera_ini_path),
        },
    }


def context_mismatches(saved, expected):
    mismatches = []
    for key in ("d455_serial", "d435i_serial", "d455_size", "d435i_size"):
        if saved.get(key) != expected.get(key):
            mismatches.append(key)
    if saved.get("core_calibration_sha256") != expected.get("core_calibration_sha256"):
        mismatches.append("core_calibration_sha256")
    return mismatches


def load_alignment(path, expected_context=None, require_validated=True):
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rotation = np.asarray(data.get("rotation_xy"), dtype=np.float64)
    translation = np.asarray(data.get("translation_xy_m"), dtype=np.float64)
    if data.get("version") != ALIGNMENT_VERSION:
        raise ValueError("相机一致性文件版本过旧，请重新完成一次验证。")
    if rotation.shape != (2, 2) or translation.shape != (2,):
        raise ValueError("相机一致性文件的矩阵尺寸无效。")
    if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
        raise ValueError("相机一致性文件包含无效数值。")
    if (not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3)
            or not np.allclose(rotation.T @ rotation, np.eye(2), atol=1e-3)):
        raise ValueError("相机一致性文件的旋转矩阵无效。")
    try:
        z_offset = float(data.get("z_offset_m"))
    except (TypeError, ValueError):
        raise ValueError("相机一致性文件的Z偏移无效。")
    if not np.isfinite(z_offset):
        raise ValueError("相机一致性文件的Z偏移无效。")
    if expected_context is not None:
        mismatches = context_mismatches(data.get("hardware_context", {}), expected_context)
        if mismatches:
            raise ValueError("相机或核心标定已改变：%s" % ", ".join(mismatches))
    if require_validated:
        validation = data.get("validation", {})
        if validation.get("status") != "passed":
            raise ValueError("相机一致性候选尚未通过独立数据验收。")
        metrics = {
            "median_m": validation.get("median_m", float("inf")),
            "max_m": validation.get("max_m", float("inf")),
        }
        if not validation_passes(metrics):
            raise ValueError("相机一致性文件记录的独立验收残差超限。")
    return data
