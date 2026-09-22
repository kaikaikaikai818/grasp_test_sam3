import sys
import types
import unittest
from pathlib import Path

import torch
from torch import nn


VENDOR = Path(__file__).resolve().parents[1] / "third_party_promptgd"
sys.path.insert(0, str(VENDOR))
sys.modules.setdefault("clip", types.ModuleType("clip"))

from inference.models.grconvnet3_CLIP import ClipModel


class _FakeClip(nn.Module):
    def encode_text(self, tokens):
        if tokens.ndim != 2:
            raise AssertionError("CLIP token input must keep its batch dimension")
        return torch.ones((tokens.shape[0], 512))


class PromptTokenTests(unittest.TestCase):
    def test_one_word_prompt_keeps_two_dimensional_token_batch(self):
        encoder = ClipModel.__new__(ClipModel)
        nn.Module.__init__(encoder)
        encoder.model = _FakeClip()
        encoder.fc = nn.Linear(512, 128)

        result = encoder(torch.zeros((1, 1, 77), dtype=torch.long))

        self.assertEqual(tuple(result.shape), (1, 128))


if __name__ == "__main__":
    unittest.main()
