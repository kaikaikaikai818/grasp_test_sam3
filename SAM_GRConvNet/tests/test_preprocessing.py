import unittest

import numpy as np
import torch

from grasp_backend.preprocessing import prepare_rgb


class PreprocessingTests(unittest.TestCase):
    def test_non_square_map_restores_to_source_shape(self):
        image = np.zeros((80, 120, 3), np.uint8)
        tensor, transform = prepare_rgb(image, 224, torch.device("cpu"))
        self.assertEqual(tuple(tensor.shape), (1, 3, 224, 224))
        restored = transform.restore_map(np.ones((224, 224), np.float32))
        self.assertEqual(restored.shape, (80, 120))


if __name__ == "__main__":
    unittest.main()
