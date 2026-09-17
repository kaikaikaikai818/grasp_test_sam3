from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .detector import ClipSegDetector
from .segmenter import MobileSamSegmenter, Segmentation


@dataclass(frozen=True)
class EngineResult:
    segmentations: list[Segmentation]
    detection_seconds: float
    segmentation_seconds: float
    cuda_peak_allocated_mb: float
    cuda_peak_reserved_mb: float


class LightSamEngine:
    """Keeps CLIPSeg and MobileSAM resident for repeated frame inference."""

    def __init__(self, checkpoint: str, prompt: str, local_files_only: bool = True) -> None:
        self.prompt = prompt
        self.detector = ClipSegDetector(local_files_only=local_files_only)
        self.segmenter = MobileSamSegmenter(checkpoint=checkpoint)

    def infer(self, image_bgr: np.ndarray) -> EngineResult:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        started = time.perf_counter()
        detections = self.detector.detect(image_rgb, self.prompt)
        detection_seconds = time.perf_counter() - started
        if not detections:
            return EngineResult([], detection_seconds, 0.0, 0.0, 0.0)

        started = time.perf_counter()
        segmentations, allocated_mb, reserved_mb = self.segmenter.segment(image_rgb, detections)
        return EngineResult(
            segmentations=segmentations,
            detection_seconds=detection_seconds,
            segmentation_seconds=time.perf_counter() - started,
            cuda_peak_allocated_mb=allocated_mb,
            cuda_peak_reserved_mb=reserved_mb,
        )
