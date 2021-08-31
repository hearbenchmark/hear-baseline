"""
Tests for the baseline model
"""

import numpy as np
import torch

from hearbaseline.vggish import (
    load_model,
    get_timestamp_embeddings,
)


torch.backends.cudnn.deterministic = True


class TestEmbeddingsTimestamps:
    def setup(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model50 = load_model(hop_length=50)
        self.model100 = load_model(hop_length=100)
        self.audio = torch.rand(2, 16000, device=self.device) * 2 - 1
        self.embeddings_ct50, self.ts_ct50 = get_timestamp_embeddings(
            audio=self.audio,
            model=self.model50,
        )
        self.embeddings_ct100, self.ts_ct100 = get_timestamp_embeddings(
            audio=self.audio,
            model=self.model100,
        )

    def teardown(self):
        del self.model50
        del self.model100
        del self.audio
        del self.embeddings_ct50
        del self.ts_ct50
        del self.embeddings_ct100
        del self.ts_ct100

    def test_timestamps_spacing(self):
        # Test the spacing between the time stamp
        diff = torch.diff(self.ts_ct50)
        assert torch.all(torch.mean(diff) - 50 < 1e-5)
        diff = torch.diff(self.ts_ct100)
        assert torch.all(torch.mean(diff) - 100 < 1e-5)
