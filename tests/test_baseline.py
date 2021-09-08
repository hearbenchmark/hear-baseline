"""
Tests for the baseline model
"""

import numpy as np
import torch

from hearbaseline import (
    load_model,
    get_timestamp_embeddings,
    get_scene_embeddings,
)
from hearbaseline.util import frame_audio
import hearbaseline.naive as baseline


torch.backends.cudnn.deterministic = True


class TestEmbeddingsTimestamps:
    def setup(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = load_model()
        self.sample_rate = self.model.sample_rate
        self.audio = torch.rand(64, 96000, device=self.device) * 2 - 1
        self.embeddings_ct, self.ts_ct = get_timestamp_embeddings(
            audio=self.audio,
            model=self.model,
        )
        self.embeddings_cs = get_scene_embeddings(
            audio=self.audio,
            model=self.model,
        )

    def teardown(self):
        del self.model
        del self.audio
        del self.embeddings_ct
        del self.ts_ct
        del self.embeddings_cs

    def test_embeddings_replicability(self):
        # Test if all the embeddings are replicable
        embeddings_ct, ts_ct = get_timestamp_embeddings(
            audio=self.audio,
            model=self.model,
        )
        assert torch.allclose(self.embeddings_ct, embeddings_ct)
        assert torch.allclose(self.ts_ct, ts_ct)

        embeddings_cs = get_scene_embeddings(
            audio=self.audio,
            model=self.model,
        )
        assert torch.allclose(self.embeddings_cs, embeddings_cs)

    def test_embeddings_batched(self):
        # methodA - Pass two audios individually and get embeddings. methodB -
        # Pass the two audio in a batch and get the embeddings. All
        # corresponding embeddings by method A and method B should be similar.
        audioa = self.audio[0].unsqueeze(0)
        audiob = self.audio[1].unsqueeze(0)
        audioab = self.audio[:2]
        assert torch.all(torch.cat([audioa, audiob]) == audioab)

        embeddingsa, _ = get_timestamp_embeddings(
            audio=audioa,
            model=self.model,
        )
        embeddingsb, _ = get_timestamp_embeddings(
            audio=audiob,
            model=self.model,
        )
        embeddingsab, _ = get_timestamp_embeddings(
            audio=audioab,
            model=self.model,
        )

        assert torch.allclose(torch.cat([embeddingsa, embeddingsb]), embeddingsab)

    def test_embeddings_sliced(self):
        # Slice the audio to select every even audio in the batch. Produce the
        # embedding for this sliced audio batch. The embeddings for
        # corresponding audios should match the embeddings when the full batch
        # was passed.
        audio_sliced = self.audio[::2]

        # Ensure framing is identical [.???] -> Yes ensuring that.
        audio_sliced_framed, _ = frame_audio(
            audio_sliced,
            frame_size=4096,
            hop_size=25.0,
            sample_rate=self.sample_rate,
        )
        audio_framed, _ = frame_audio(
            self.audio,
            frame_size=4096,
            hop_size=25.0,
            sample_rate=self.sample_rate,
        )
        assert torch.all(audio_sliced_framed == audio_framed[::2])

        # Test for centered
        embeddings_sliced, _ = get_timestamp_embeddings(
            audio=audio_sliced,
            model=self.model,
        )

        assert torch.allclose(embeddings_sliced, self.embeddings_ct[::2])

    def test_embeddings_shape(self):
        # Test the embeddings shape.
        # The shape returned is (batch_size, num_frames, embedding_size). We expect
        # num_frames to be equal to the number of full audio frames that can fit into
        # the audio sample. The centered example is padded with frame_size (4096) number
        # of samples, so we don't need to subtract that in that test.
        hop_size_ms = baseline.TIMESTAMP_HOP_SIZE
        hop_size_samples = int(hop_size_ms / 1000.0 * self.sample_rate)
        assert self.embeddings_ct.shape == (
            64,
            96000 // hop_size_samples + 1,
            int(4096),
        )
        hop_size_ms = baseline.SCENE_HOP_SIZE
        hop_size_samples = int(hop_size_ms / 1000.0 * self.sample_rate)
        assert self.embeddings_cs.shape == (
            64,
            int(4096),
        )

    def test_timestamps_shape(self):
        # Make sure the timestamps have the correct shape
        assert self.embeddings_ct.shape[:2] == self.ts_ct.shape

    def test_embeddings_nan(self):
        # Test for null values in the embeddings.
        assert not torch.any(torch.isnan(self.embeddings_ct))
        assert not torch.any(torch.isnan(self.embeddings_cs))

    def test_embeddings_type(self):
        # Test the data type of the embeddings.
        assert self.embeddings_ct.dtype == torch.float32
        assert self.embeddings_cs.dtype == torch.float32

    def test_timestamps_begin(self):
        # Test the beginning of the time stamp. Should be zero for all
        # timestamps in the batch
        assert torch.all(self.ts_ct[:, 0] == 0)

    def test_timestamps_spacing(self):
        # Test the spacing between the time stamp
        diff = torch.diff(self.ts_ct)
        assert torch.all(torch.mean(diff) - self.ts_ct[:, 1] < 1e-5)


class TestModel:
    def setup(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = load_model().to(device)
        self.frames = torch.rand(512, self.model.n_fft, device=device) * 2 - 1

    def teardown(self):
        del self.model
        del self.frames

    def test_model_attributes(self):
        # Each model should have sample_rate and embedding size information
        assert hasattr(self.model, "sample_rate")
        assert hasattr(self.model, "embedding_size")

    def test_model_sliced(self):
        frames_sliced = self.frames[::2]
        assert torch.allclose(frames_sliced[0], self.frames[0])
        assert torch.allclose(frames_sliced[1], self.frames[2])
        assert torch.allclose(frames_sliced, self.frames[::2])

        outputs = self.model(self.frames)
        outputs_sliced = self.model(frames_sliced)

        assert torch.allclose(outputs_sliced[0], outputs[0])
        assert torch.allclose(outputs_sliced[1], outputs[2])
        assert torch.allclose(outputs_sliced, outputs[::2])


class TestFraming:
    def test_frame_audio(self):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        sr = 44100
        num_audio = 16
        duration = 1.1
        frame_size = 4096
        hop_size_ms = 25.0

        audio = torch.rand((num_audio, int(sr * duration)), device=device)
        frames, timestamps = frame_audio(
            audio, frame_size=frame_size, hop_size=hop_size_ms, sample_rate=sr
        )

        hop_size_samples = hop_size_ms / 1000.0 * sr
        expected_frames = int(sr * duration / hop_size_samples) + 1
        expected_frames_shape = (num_audio, expected_frames, frame_size)
        expected_timestamps = np.arange(0, expected_frames)
        expected_timestamps = expected_timestamps * hop_size_ms

        assert expected_frames_shape == frames.shape
        assert np.allclose(expected_timestamps, timestamps.detach().cpu().numpy())
