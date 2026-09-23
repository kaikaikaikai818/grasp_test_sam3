from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .depth_position import estimate_mask_position
from .engine import EngineResult, LightSamEngine
from .pipeline import COLORS
from .realsense_camera import CameraFrame, RealSenseCamera, find_required_devices


def _render_and_save(
    role: str,
    frame: CameraFrame,
    result: EngineResult,
    output_root: Path,
    prompt: str,
) -> np.ndarray:
    overlay = frame.color_bgr.copy()
    objects = []
    output_dir = output_root / role
    details_dir = output_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    for old_mask in details_dir.glob("sam_mask_*.png"):
        old_mask.unlink()

    for index, segmentation in enumerate(result.segmentations, start=1):
        color = COLORS[(index - 1) % len(COLORS)]
        position = estimate_mask_position(
            segmentation.mask,
            frame.aligned_depth,
            frame.depth_scale,
            frame.intrinsics,
        )
        overlay[segmentation.mask] = color
        x1, y1, x2, y2 = segmentation.detection.box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        if position.xyz_m is None:
            label = f"#{index} C{segmentation.detection.score:.2f} XYZ invalid"
            xyz_json = None
        else:
            x, y, z = position.xyz_m
            label = (
                f"#{index} C{segmentation.detection.score:.2f} D{position.median_depth_m:.3f}m "
                f"XYZ({x:.3f},{y:.3f},{z:.3f})"
            )
            xyz_json = [x, y, z]
        cv2.putText(
            overlay,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        mask_name = f"sam_mask_{index:03d}.png"
        cv2.imwrite(str(details_dir / mask_name), segmentation.mask.astype(np.uint8) * 255)
        objects.append(
            {
                "id": index,
                "box": list(segmentation.detection.box),
                "clipseg_score": segmentation.detection.score,
                "sam_score": segmentation.sam_score,
                "xyz_m": xyz_json,
                "median_depth_m": position.median_depth_m,
                "valid_depth_points": position.valid_depth_points,
                "mask": mask_name,
            }
        )

    visualization = cv2.addWeighted(frame.color_bgr, 0.65, overlay, 0.35, 0)
    if not objects:
        cv2.putText(
            visualization,
            "No matching target",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "role": "global_search" if role == "d455" else "wrist_refinement",
        "camera_model": frame.device.model,
        "camera_serial": frame.device.serial,
        "usb_type": frame.device.usb_type,
        "frame_timestamp_ms": frame.timestamp_ms,
        "coordinate_system": {
            "name": f"{role}_optical_frame",
            "unit": "meter",
            "axes": {"x": "right", "y": "down", "z": "forward"},
        },
        "prompt": prompt,
        "object_count": len(objects),
        "objects": objects,
        "runtime": {
            "clipseg_cpu_seconds": result.detection_seconds,
            "mobilesam_cuda_seconds": result.segmentation_seconds,
            "cuda_peak_allocated_mb": result.cuda_peak_allocated_mb,
            "cuda_peak_reserved_mb": result.cuda_peak_reserved_mb,
        },
    }
    cv2.imwrite(str(output_dir / "sam_result.jpg"), visualization)
    temporary = details_dir / "result.json.tmp"
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(details_dir / "result.json")
    return visualization


def run_realsense(
    checkpoint: str,
    prompt: str,
    output_root: str,
    d455_serial: str | None = None,
    d435i_serial: str | None = None,
) -> None:
    devices = find_required_devices(d455_serial, d435i_serial)
    cameras: dict[str, RealSenseCamera] = {}
    try:
        cameras["d455"] = RealSenseCamera(devices["d455"])
        cameras["d435i"] = RealSenseCamera(devices["d435i"])
        engine = LightSamEngine(checkpoint=checkpoint, prompt=prompt, local_files_only=True)
        root = Path(output_root)

        while True:
            views = {}
            for role in ("d455", "d435i"):
                frame = cameras[role].capture()
                try:
                    result = engine.infer(frame.color_bgr)
                    views[role] = _render_and_save(role, frame, result, root, prompt)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise RuntimeError("CUDA 显存不足。请关闭占用显卡的软件后重试。") from exc
                    views[role] = frame.color_bgr.copy()
                    cv2.putText(
                        views[role], "Inference failed - preview continues", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA,
                    )
            cv2.imshow(
                f"D455 - Global Search [{devices['d455'].serial}]", views["d455"]
            )
            cv2.imshow(
                f"D435i - Wrist Refinement [{devices['d435i'].serial}]", views["d435i"]
            )
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        for camera in cameras.values():
            camera.stop()
        cv2.destroyAllWindows()
