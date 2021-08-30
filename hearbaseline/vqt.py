"""
variable-Q transform spectrogram (60 bins/octave, ~10ms step)

Based upon DCASE 2016 Task 2 baseline.
The scene embeddings will be an average, i.e. only 84 dimensions.
"""

import math
import numpy as np
from typing import Tuple

import librosa
import torch
from torch import Tensor


class VQT(torch.nn.Module):
    # Not really a torch module

    # sample rate and embedding sizes are required model attributes for the HEAR API
    sample_rate = 44100
    # dcase baseline was 10 ms (441), but for librosa hop_length must be
    # a positive integer multiple of 2^8 for 9-octave CQT/VQT
    hop_length = 512
    gamma = 30
    bins_per_octave = 60
    fmin = 27.5
    noctaves = int(math.log2(sample_rate / 2 / fmin))

    embedding_size = bins_per_octave * noctaves
    scene_embedding_size = embedding_size
    timestamp_embedding_size = embedding_size

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        embeddings = []
        for i in range(x.shape[0]):
            embedding = np.abs(
                librosa.vqt(
                    x[i].detach().cpu().numpy(),
                    sr=self.sample_rate,
                    hop_length=self.hop_length,
                    gamma=self.gamma,
                    n_bins=self.embedding_size,
                    bins_per_octave=self.bins_per_octave,
                    fmin=self.fmin,
                )
            ).T
            embedding = torch.tensor(embedding, device=x.device)
            embeddings.append(embedding)
        embeddings = torch.stack(embeddings)
        return embeddings


def load_model(model_file_path: str = "") -> torch.nn.Module:
    """
    Returns a torch.nn.Module that produces embeddings for audio.

    Args:
        model_file_path: Ignored.
    Returns:
        Model
    """

    return VQT()


def get_timestamp_embeddings(
    audio: Tensor,
    model: torch.nn.Module,
) -> Tuple[Tensor, Tensor]:
    """
    This function returns embeddings at regular intervals centered at timestamps. Both
    the embeddings and corresponding timestamps (in milliseconds) are returned.

    Args:
        audio: n_sounds x n_samples of mono audio in the range [-1, 1].
        model: Loaded model.

    Returns:
        - Tensor: embeddings, A float32 Tensor with shape (n_sounds, n_timestamps,
            model.timestamp_embedding_size).
        - Tensor: timestamps, Centered timestamps in milliseconds corresponding
            to each embedding in the output. Shape: (n_sounds, n_timestamps).
    """

    # Assert audio is of correct shape
    if audio.ndim != 2:
        raise ValueError(
            "audio input tensor must be 2D with shape (n_sounds, num_samples)"
        )

    # Make sure the correct model type was passed in
    if not isinstance(model, VQT):
        raise ValueError(f"Model must be an instance of {VQT.__name__}")

    # Send the model to the same device that the audio tensor is on.
    # model = model.to(audio.device)

    # Put the model into eval mode, and not computing gradients while in inference.
    # Iterate over all batches and accumulate the embeddings for each frame.
    with torch.no_grad():
        embeddings = model(audio)

    ntimestamps = audio.shape[1] // model.hop_length + 1
    hop_length_ms = 1000 * model.hop_length / model.sample_rate
    timestamps = torch.arange(0, ntimestamps * hop_length_ms, hop_length_ms)
    assert len(timestamps) == ntimestamps
    timestamps = timestamps.expand((embeddings.shape[0], timestamps.shape[0]))
    assert timestamps.shape[1] == embeddings.shape[1]

    return embeddings, timestamps


# TODO: There must be a better way to do scene embeddings,
# e.g. just truncating / padding the audio to 2 seconds
# and concatenating a subset of the embeddings.
def get_scene_embeddings(
    audio: Tensor,
    model: torch.nn.Module,
) -> Tensor:
    """
    This function returns a single embedding for each audio clip. In this baseline
    implementation we simply summarize the temporal embeddings from
    get_timestamp_embeddings() using torch.mean().

    Args:
        audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in
            a batch will be padded/trimmed to the same length.
        model: Loaded model.

    Returns:
        - embeddings, A float32 Tensor with shape
            (n_sounds, model.scene_embedding_size).
    """
    embeddings, _ = get_timestamp_embeddings(audio, model)
    embeddings = torch.mean(embeddings, dim=1)
    return embeddings
