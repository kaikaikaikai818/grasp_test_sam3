"""Text-prompted tool detector backed by the parent MobileSAM project."""

from pathlib import Path
import sys

import cv2
import numpy as np


MOBILE_SAM_ROOT = Path(__file__).resolve().parents[3]
if str(MOBILE_SAM_ROOT) not in sys.path:
    sys.path.insert(0, str(MOBILE_SAM_ROOT))

from light_sam.engine import LightSamEngine


class SamToolDetector:
    """Keep the models resident and expose the result shape used by the UR5 code."""

    def __init__(self, prompt, checkpoint=None, min_depth_m=0.15, max_depth_m=3.0):
        self.prompt = prompt.strip()
        if not self.prompt:
            raise ValueError("工具文字提示不能为空")
        checkpoint = checkpoint or MOBILE_SAM_ROOT / "weights" / "mobile_sam.pt"
        self.engine = LightSamEngine(str(checkpoint), self.prompt, local_files_only=True)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)

    def _to_result(self, segmentation, depth_raw, depth_scale):
        mask = segmentation.mask.astype(bool)
        eroded = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
        if not eroded.any():
            eroded = mask
        depth_m = depth_raw.astype(np.float32) * float(depth_scale)
        valid = eroded & (depth_m >= self.min_depth_m) & (depth_m <= self.max_depth_m)
        values = depth_m[valid]
        if values.size == 0:
            z_mm = None
        else:
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            tolerance = max(0.03, 3.0 * 1.4826 * mad)
            valid &= np.abs(depth_m - median) <= tolerance
            values = depth_m[valid]
            z_mm = float(np.median(values) * 1000.0) if values.size else None

        pixels = np.argwhere(valid if valid.any() else eroded)
        if pixels.size:
            v, u = np.median(pixels, axis=0).astype(int)
        else:
            x1, y1, x2, y2 = segmentation.detection.box
            u, v = (x1 + x2) // 2, (y1 + y2) // 2

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        angle = 0.0
        area = float(mask.sum())
        if contours:
            contour = max(contours, key=cv2.contourArea)
            (_, _), (width, height), raw_angle = cv2.minAreaRect(contour)
            angle = float(raw_angle + (90.0 if width < height else 0.0)) % 180.0
        return {
            "center": (int(u), int(v)),
            "z_mm": z_mm,
            "area": area,
            "mask": mask,
            "box": tuple(int(v) for v in segmentation.detection.box),
            "score": float(segmentation.detection.score),
            "sam_score": float(segmentation.sam_score),
            "angle_deg": angle,
            "prompt": self.prompt,
            "valid_depth_points": int(np.count_nonzero(valid)),
        }

    def detect_all(self, bgr, depth_raw, depth_scale):
        inference = self.engine.infer(bgr)
        results = [self._to_result(item, depth_raw, depth_scale)
                   for item in inference.segmentations]
        return sorted(results, key=lambda item: item["score"], reverse=True)

    def detect(self, bgr, depth_raw, depth_scale):
        results = self.detect_all(bgr, depth_raw, depth_scale)
        return results[0] if results else None

    @staticmethod
    def draw(bgr, result, color=(0, 255, 0)):
        image = bgr.copy()
        if result is None:
            return image
        mask = result["mask"]
        tint = image.copy()
        tint[mask] = color
        image = cv2.addWeighted(image, 0.65, tint, 0.35, 0)
        x1, y1, x2, y2 = result["box"]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        u, v = result["center"]
        cv2.circle(image, (u, v), 5, (0, 0, 255), -1)
        depth = "invalid" if result["z_mm"] is None else "%.1fmm" % result["z_mm"]
        text = "%s %.2f %s angle=%.1f" % (
            result["prompt"], result["score"], depth, result["angle_deg"])
        cv2.putText(image, text, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)
        return image
