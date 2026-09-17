"""Planar relative alignment between D455 and D435i base-frame estimates."""

import json
from pathlib import Path

import numpy as np


ALIGNMENT_VERSION = 1


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
        "rotation_xy": rotation.tolist(),
        "translation_xy_m": translation.tolist(),
        "z_offset_m": z_offset,
        "fit_residual_m": residuals.tolist(),
        "fit_residual_median_m": float(np.median(residuals)),
        "fit_residual_max_m": float(np.max(residuals)),
    }


def apply_alignment_array(points, rotation, translation, z_offset):
    points = np.asarray(points, dtype=np.float64)
    output = points.copy()
    output[:, :2] = (np.asarray(rotation) @ points[:, :2].T).T + translation
    output[:, 2] += float(z_offset)
    return output


def apply_alignment(point_xyz, alignment):
    point = np.asarray(point_xyz, dtype=np.float64).reshape(1, 3)
    aligned = apply_alignment_array(
        point,
        np.asarray(alignment["rotation_xy"], dtype=np.float64),
        np.asarray(alignment["translation_xy_m"], dtype=np.float64),
        float(alignment["z_offset_m"]),
    )
    return tuple(float(value) for value in aligned[0])


def load_alignment(path):
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rotation = np.asarray(data.get("rotation_xy"), dtype=np.float64)
    translation = np.asarray(data.get("translation_xy_m"), dtype=np.float64)
    if data.get("version") != ALIGNMENT_VERSION:
        raise ValueError("不支持的相机校正文件版本。")
    if rotation.shape != (2, 2) or translation.shape != (2,):
        raise ValueError("相机校正文件的矩阵尺寸无效。")
    if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
        raise ValueError("相机校正文件包含无效数值。")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
        raise ValueError("相机校正旋转矩阵无效。")
    return data
