import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bsp.camera_bsp.yoloe_tool_detect import YoloEEngine


class FakeModel:
    def predict(self, image, **kwargs):
        mask = torch.zeros((1, 32, 40))
        mask[0, 9:19, 12:29] = 1
        box = SimpleNamespace(xyxy=torch.tensor([[12., 9., 29., 19.]]),
                              conf=torch.tensor(.8))
        return [SimpleNamespace(boxes=[box], masks=SimpleNamespace(data=mask))]


class YoloEAdapterTests(unittest.TestCase):
    def test_inference_returns_existing_detector_shape(self):
        engine = YoloEEngine.__new__(YoloEEngine)
        engine.device = "cpu"
        engine.model = FakeModel()
        result = engine.infer(np.zeros((32, 40, 3), np.uint8))
        self.assertEqual(len(result.segmentations), 1)
        segmentation = result.segmentations[0]
        self.assertEqual(segmentation.detection.box, (12, 9, 29, 19))
        self.assertTrue(segmentation.mask[13, 18])
        self.assertGreater(result.detection_seconds, 0)


if __name__ == "__main__":
    unittest.main()
