from __future__ import annotations

import cv2
import numpy as np

from .fusion import FusionResult, select_masked_grasp
from .types import PredictionResult


def _principal_axis(mask: np.ndarray) -> tuple[np.ndarray, float]:
    ys, xs = np.nonzero(mask)
    points = np.column_stack((xs, ys)).astype(np.float64)
    if len(points) < 2:
        return np.array((1.0, 0.0)), 0.0
    covariance = np.cov(points, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)
    major = vectors[:, order[-1]]
    major /= max(np.linalg.norm(major), 1e-9)
    major_value = max(float(values[order[-1]]), 1e-9)
    minor_value = max(float(values[order[-2]]), 0.0)
    anisotropy = float(np.clip(1.0 - minor_value / major_value, 0.0, 1.0))
    return major, anisotropy


def _choose_center(mask: np.ndarray, distance: np.ndarray, major: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    points = np.column_stack((xs, ys)).astype(np.float64)
    projection = points @ major
    low, high = np.quantile(projection, (0.05, 0.95))
    if high - low < 2:
        index = int(np.argmax(distance[ys, xs]))
        return int(xs[index]), int(ys[index])

    edges = np.linspace(low, high, 22)
    centers: list[tuple[int, int]] = []
    radii: list[float] = []
    for start, end in zip(edges[:-1], edges[1:]):
        selected = (projection >= start) & (projection < end)
        if not selected.any():
            centers.append((-1, -1))
            radii.append(0.0)
            continue
        selected_indices = np.flatnonzero(selected)
        local_distances = distance[ys[selected], xs[selected]]
        point_index = selected_indices[int(np.argmax(local_distances))]
        centers.append((int(xs[point_index]), int(ys[point_index])))
        radii.append(float(distance[ys[point_index], xs[point_index]]))

    valid_indices = [index for index in range(4, 17) if radii[index] > 0]
    if not valid_indices:
        index = int(np.argmax(distance[ys, xs]))
        return int(xs[index]), int(ys[index])

    left_radius = np.median([radii[index] for index in valid_indices[:3]])
    right_radius = np.median([radii[index] for index in valid_indices[-3:]])
    if left_radius < right_radius * 0.8:
        target = 6
    elif right_radius < left_radius * 0.8:
        target = 14
    else:
        target = 10
    nearby = [index for index in valid_indices if abs(index - target) <= 2]
    chosen = max(nearby or valid_indices, key=lambda index: radii[index])
    return centers[chosen]


def geometry_maps(mask: np.ndarray, opening_scale: float = 1.15) -> dict[str, np.ndarray]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    shape = binary.shape
    quality = np.zeros(shape, np.float32)
    angle = np.zeros(shape, np.float32)
    width = np.zeros(shape, np.float32)
    strength = np.zeros(shape, np.float32)
    if binary.sum() == 0:
        return {"quality": quality, "angle": angle, "width": width,
                "angle_strength": strength}

    distance = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    major, anisotropy = _principal_axis(binary)
    center_x, center_y = _choose_center(binary, distance, major)

    closing = np.array((-major[1], major[0]))
    theta = np.arctan2(-closing[1], closing[0])
    theta = (theta + np.pi / 2) % np.pi - np.pi / 2
    local_radius = max(float(distance[center_y, center_x]), 1.0)
    sigma = max(local_radius * 1.5, 4.0)
    grid_y, grid_x = np.indices(shape)
    confidence = float(0.60 + 0.30 * anisotropy)
    quality[:] = (
        confidence
        * np.exp(-((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2) / (2 * sigma**2))
        * binary
    )
    angle[binary] = theta
    width[binary] = 2.0 * distance[binary] * float(opening_scale)
    strength[binary] = max(0.10, anisotropy)
    return {"quality": quality, "angle": angle, "width": width,
            "angle_strength": strength}


class GeometryGraspPredictor:
    backend_name = "mask geometry baseline"

    def __init__(self, opening_scale: float = 1.15):
        if not 1.0 <= opening_scale <= 1.45:
            raise ValueError("opening_scale must be between 1.0 and 1.45")
        self.opening_scale = float(opening_scale)

    def predict(self, image, text: str, masks) -> PredictionResult:
        height, width = image.shape[:2]
        combined = {
            "quality": np.zeros((height, width), np.float32),
            "angle": np.zeros((height, width), np.float32),
            "width": np.zeros((height, width), np.float32),
            "angle_strength": np.zeros((height, width), np.float32),
        }
        grasps: list[FusionResult] = []
        for raw_mask in masks:
            mask = np.asarray(raw_mask, dtype=bool)
            if mask.shape != (height, width):
                raise ValueError("mask and image dimensions differ")
            maps = geometry_maps(mask, self.opening_scale)
            result = select_masked_grasp(maps, mask)
            grasps.append(result)
            replace = maps["quality"] > combined["quality"]
            for name in combined:
                combined[name][replace] = maps[name][replace]
        return PredictionResult(
            combined,
            grasps,
            {"type": "mask_geometry", "image_height": height, "image_width": width},
            self.backend_name,
        )
