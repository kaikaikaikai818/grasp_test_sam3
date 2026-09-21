"""Optional local, open-vocabulary instance segmenter with the SAM result shape.

The text encoder remains available at runtime: an exported YOLOE model with
classes baked in would not meet the project's new-tool requirement.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    point: tuple[int, int]
    score: float
    area: float


@dataclass(frozen=True)
class Segmentation:
    detection: Detection
    mask: np.ndarray
    sam_score: float


@dataclass(frozen=True)
class EngineResult:
    segmentations: list[Segmentation]
    detection_seconds: float
    segmentation_seconds: float
    cuda_peak_allocated_mb: float
    cuda_peak_reserved_mb: float


class YoloEEngine:
    def __init__(self, prompt: str, checkpoint: str = "yoloe-26s-seg.pt",
                 local_files_only: bool = True) -> None:
        if not prompt.strip():
            raise ValueError("tool prompt cannot be empty")
        if local_files_only and not Path(checkpoint).is_file():
            raise FileNotFoundError(
                f"YOLOE weights not found: {checkpoint}. Download the checkpoint and "
                "its text encoder before offline use.")
        try:
            from ultralytics import YOLOE
        except ImportError as exc:
            raise RuntimeError("YOLOE requires the optional ultralytics package") from exc
        self.prompt = prompt.strip()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLOE(checkpoint)
        self.model.set_classes([self.prompt])

    def infer(self, image_bgr: np.ndarray) -> EngineResult:
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = self.model.predict(image_bgr, device=self.device, verbose=False)[0]
        elapsed = time.perf_counter() - started
        segments = []
        height, width = image_bgr.shape[:2]
        if result.boxes is not None and result.masks is not None:
            polygons = getattr(result.masks, "xy", None)
            for index, (box, mask_tensor) in enumerate(zip(result.boxes, result.masks.data)):
                if polygons is not None:
                    mask = np.zeros((height, width), dtype=np.uint8)
                    polygon = np.asarray(polygons[index], dtype=np.int32)
                    if polygon.shape[0] >= 3:
                        cv2.fillPoly(mask, [polygon], 1)
                    mask = mask.astype(bool)
                else:
                    mask = mask_tensor.cpu().numpy() > 0.5
                    if mask.shape != (height, width):
                        mask = cv2.resize(mask.astype(np.uint8), (width, height),
                                          interpolation=cv2.INTER_NEAREST).astype(bool)
                if not mask.any():
                    continue
                x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
                distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
                _, _, _, point = cv2.minMaxLoc(distance)
                detection = Detection((x1, y1, x2, y2), point,
                                      float(box.conf.item()), float(mask.sum()))
                segments.append(Segmentation(detection, mask, float(box.conf.item())))
        allocated = (torch.cuda.max_memory_allocated() / 1024**2
                     if self.device.startswith("cuda") else 0.0)
        reserved = (torch.cuda.max_memory_reserved() / 1024**2
                    if self.device.startswith("cuda") else 0.0)
        return EngineResult(segments, elapsed, 0.0, allocated, reserved)
