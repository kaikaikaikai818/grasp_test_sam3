from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from mobile_sam import SamPredictor, sam_model_registry

from .detector import Detection


@dataclass(frozen=True)
class Segmentation:
    detection: Detection
    mask: np.ndarray
    sam_score: float


class MobileSamSegmenter:
    def __init__(self, checkpoint: str, model_type: str = "vit_t", device: str = "cuda:0") -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "MobileSAM 需要 CUDA，但当前环境无法使用 CUDA。\n"
                f"PyTorch 版本: {torch.__version__}\n"
                f"PyTorch CUDA 构建版本: {torch.version.cuda}\n"
                "请确认 VS Code 已选择项目的 .venv。"
            )
        self.device = torch.device(device)
        self.model = sam_model_registry[model_type](checkpoint=checkpoint)
        self.model.to(device=self.device)
        self.model.eval()
        self.predictor = SamPredictor(self.model)

    def segment(self, image_rgb: np.ndarray, detections: list[Detection]) -> tuple[list[Segmentation], float, float]:
        torch.cuda.reset_peak_memory_stats(self.device)
        try:
            self.predictor.set_image(image_rgb)
            results: list[Segmentation] = []
            for detection in detections:
                with torch.inference_mode():
                    masks, scores, _ = self.predictor.predict(
                        point_coords=np.array([detection.point]),
                        point_labels=np.array([1]),
                        box=np.array(detection.box),
                        multimask_output=True,
                    )
                best_index = int(np.argmax(scores))
                results.append(Segmentation(detection, masks[best_index], float(scores[best_index])))
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "MobileSAM 运行时 CUDA 显存不足。请关闭占用显卡的软件后重试；"
                "如果仍然失败，请减少 --max-objects。"
            ) from exc
        allocated_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2
        reserved_mb = torch.cuda.max_memory_reserved(self.device) / 1024**2
        return results, allocated_mb, reserved_mb
