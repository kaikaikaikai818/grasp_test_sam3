from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from skimage.transform import resize


@dataclass(frozen=True)
class SquareTransform:
    source_height: int
    source_width: int
    square_side: int
    pad_left: int
    pad_top: int
    input_size: int

    def restore_map(self, value: np.ndarray) -> np.ndarray:
        square = cv2.resize(
            value, (self.square_side, self.square_side), interpolation=cv2.INTER_LINEAR
        )
        return square[
            self.pad_top:self.pad_top + self.source_height,
            self.pad_left:self.pad_left + self.source_width,
        ]

    def as_dict(self) -> dict[str, int]:
        return {
            "source_height": self.source_height,
            "source_width": self.source_width,
            "square_side": self.square_side,
            "pad_left": self.pad_left,
            "pad_top": self.pad_top,
            "input_size": self.input_size,
        }


def prepare_rgb(image_bgr: np.ndarray, input_size: int, device: torch.device):
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image must be a non-empty BGR image")
    height, width = image_bgr.shape[:2]
    side = max(height, width)
    left, top = (side - width) // 2, (side - height) // 2
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    square = cv2.copyMakeBorder(
        rgb, top, side - height - top, left, side - width - left,
        cv2.BORDER_REPLICATE,
    )
    scaled = resize(square, (input_size, input_size), preserve_range=True).astype(np.uint8)
    scaled = scaled.astype(np.float32) / 255.0
    scaled -= scaled.mean()
    tensor = torch.from_numpy(scaled.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    transform = SquareTransform(height, width, side, left, top, input_size)
    return tensor, transform
