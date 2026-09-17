from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from .detector import ClipSegDetector
from .segmenter import MobileSamSegmenter


COLORS = [(0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255), (0, 255, 255), (255, 0, 0)]


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"无法保存图片: {path}")


def run_pipeline(image_path: str, prompt: str, output_dir: str = ".",
                 checkpoint: str = "weights/mobile_sam.pt", threshold: float = 0.3,
                 min_area: float = 100.0, min_score: float = 0.35, max_objects: int = 10,
                 morphology_kernel: int = 5, local_files_only: bool = False) -> dict:
    started_at = time.perf_counter()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    details_path = output_path / "details"
    if details_path.exists():
        shutil.rmtree(details_path)
    details_path.mkdir()
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"找不到图片: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    detector = ClipSegDetector(threshold=threshold, min_area=min_area, min_score=min_score,
                               max_objects=max_objects,
                               morphology_kernel=morphology_kernel, local_files_only=local_files_only)
    detection_started = time.perf_counter()
    detections = detector.detect(image_rgb, prompt)
    detection_seconds = time.perf_counter() - detection_started
    if not detections:
        raise RuntimeError("CLIPSeg 没有找到目标，请调整提示词或降低 --threshold。")

    segmentation_started = time.perf_counter()
    segmenter = MobileSamSegmenter(checkpoint=checkpoint)
    segmentations, allocated_mb, reserved_mb = segmenter.segment(image_rgb, detections)
    segmentation_seconds = time.perf_counter() - segmentation_started

    overlay = image_bgr.copy()
    metadata_objects = []
    for index, segmentation in enumerate(segmentations, start=1):
        color = COLORS[(index - 1) % len(COLORS)]
        overlay[segmentation.mask] = color
        x1, y1, x2, y2 = segmentation.detection.box
        px, py = segmentation.detection.point
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
        cv2.circle(overlay, (px, py), 6, (0, 0, 255), -1)
        cv2.putText(overlay, f"#{index} {segmentation.sam_score:.3f}", (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        mask_name = f"sam_mask_{index:03d}.png"
        _write_image(details_path / mask_name, segmentation.mask.astype(np.uint8) * 255)
        metadata_objects.append({
            "id": index, "box": [x1, y1, x2, y2], "point": [px, py],
            "clipseg_score": segmentation.detection.score,
            "sam_score": segmentation.sam_score,
            "contour_area": segmentation.detection.area, "mask": f"details/{mask_name}",
        })

    result = cv2.addWeighted(image_bgr, 0.65, overlay, 0.35, 0)
    _write_image(output_path / "sam_result.jpg", result)
    metadata = {
        "image": str(Path(image_path).resolve()),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "image_size": {"width": width, "height": height},
        "prompt": prompt, "threshold": threshold, "min_score": min_score,
        "object_count": len(metadata_objects), "objects": metadata_objects,
        "result": "sam_result.jpg",
        "runtime": {
            "clipseg_cpu_seconds": detection_seconds,
            "mobilesam_cuda_seconds": segmentation_seconds,
            "total_seconds": time.perf_counter() - started_at,
            "cuda_peak_allocated_mb": allocated_mb,
            "cuda_peak_reserved_mb": reserved_mb,
        },
        "environment": {
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    (details_path / "sam_result.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata
