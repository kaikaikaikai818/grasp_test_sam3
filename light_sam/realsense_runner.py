from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .depth_position import estimate_mask_position
from .detector import Detection
from .engine import EngineResult, LightSamEngine
from .pipeline import COLORS
from .realsense_camera import CameraFrame, RealSenseCamera, find_required_devices
from .segmenter import Segmentation
from .screwdriver_grasp import plan_screwdriver_grasp


def restore_roi_result(
    result: EngineResult,
    full_shape: tuple[int, int],
    roi: tuple[int, int, int, int],
) -> EngineResult:
    """Restore crop-local detections and masks to full-frame pixel coordinates."""
    height, width = full_shape
    x1, y1, x2, y2 = [int(value) for value in roi]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"D455_ROI {roi} 超出图像范围 {width}x{height}。")
    restored = []
    for item in result.segmentations:
        local_height, local_width = item.mask.shape
        if (local_height, local_width) != (y2 - y1, x2 - x1):
            raise ValueError("D455 裁剪推理返回了错误尺寸的 mask。")
        full_mask = np.zeros((height, width), dtype=bool)
        full_mask[y1:y2, x1:x2] = item.mask
        box_x1, box_y1, box_x2, box_y2 = item.detection.box
        point_x, point_y = item.detection.point
        detection = Detection(
            (box_x1 + x1, box_y1 + y1, box_x2 + x1, box_y2 + y1),
            (point_x + x1, point_y + y1),
            item.detection.score,
            item.detection.area,
        )
        restored.append(Segmentation(detection, full_mask, item.sam_score))
    return EngineResult(
        restored,
        result.detection_seconds,
        result.segmentation_seconds,
        result.cuda_peak_allocated_mb,
        result.cuda_peak_reserved_mb,
    )


def _render_and_save(
    role: str,
    frame: CameraFrame,
    result: EngineResult,
    output_root: Path,
    prompt: str,
    roi: tuple[int, int, int, int] | None = None,
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
        grasp, grasp_reason = plan_screwdriver_grasp(
            segmentation.mask, frame.aligned_depth, frame.depth_scale
        )
        if grasp is not None:
            corners = np.rint(grasp.corners_xy).astype(np.int32)
            cv2.polylines(overlay, [corners], True, (0, 230, 255), 3)
            cv2.circle(overlay, grasp.center_xy, 5, (0, 0, 255), -1)
            grasp_y = min(frame.color_bgr.shape[0] - 12, y2 + 22)
            cv2.putText(
                overlay,
                f"GRASP handle A={grasp.angle_deg:.1f} W={grasp.opening_width_px:.0f}px",
                (x1, grasp_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (0, 230, 255), 2, cv2.LINE_AA,
            )
            grasp_json = {
                "accepted": True,
                "candidate": grasp.as_dict(),
                "rejection_reason": None,
            }
        else:
            grasp_json = {
                "accepted": False,
                "candidate": None,
                "rejection_reason": grasp_reason,
            }
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
                "grasp": grasp_json,
            }
        )

    visualization = cv2.addWeighted(frame.color_bgr, 0.65, overlay, 0.35, 0)
    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(visualization, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.putText(visualization, "D455 search area", (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2, cv2.LINE_AA)
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
        "search_roi_xyxy": list(roi) if roi is not None else None,
        "object_count": len(objects),
        "accepted_grasp_count": sum(
            1 for item in objects if item["grasp"]["accepted"]
        ),
        "objects": objects,
        "robot_motion_authorized": False,
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
    d455_roi: tuple[int, int, int, int] | None = None,
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
                    roi = d455_roi if role == "d455" else None
                    if roi is None:
                        result = engine.infer(frame.color_bgr)
                    else:
                        x1, y1, x2, y2 = roi
                        crop = frame.color_bgr[y1:y2, x1:x2]
                        if crop.size == 0:
                            raise ValueError(f"D455_ROI {roi} 为空，请检查坐标。")
                        result = restore_roi_result(
                            engine.infer(crop), frame.color_bgr.shape[:2], roi
                        )
                    views[role] = _render_and_save(
                        role, frame, result, root, prompt, roi=roi
                    )
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
