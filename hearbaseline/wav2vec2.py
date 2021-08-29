"""
wav2vec2 model for HEAR 2021 NeurIPS competition.

Adapted from
https://colab.research.google.com/drive/17Hu1pxqhfMisjkSgmM2CnZxfqDyn2hSY?usp=sharing
"""

from typing import Tuple

import torch

# from speechbrain.lobes.models.fairseq_wav2vec import FairseqWav2Vec2
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
from torch import Tensor

# HuggingFace model hub
model_hub = "facebook/wav2vec2-base-960h"

# Faiseq model url
# model_url = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt"


def load_model(model_file_path: str = "") -> torch.nn.Module:
    """
    Returns a torch.nn.Module that produces embeddings for audio.

    Args:
        model_file_path: Ignored.
    Returns:
        Model
    """
    # model_fairseq = FairseqWav2Vec2(model_url, save_path="pretrained/local_model.pt")
    model_huggingface = HuggingFaceWav2Vec2(model_hub, save_path="pretrained/")
    model = model_huggingface
    if torch.cuda.is_available():
        model.cuda()

    # sample rate and embedding sizes are required model attributes for the HEAR API
    model.sample_rate = 16000
    model.embedding_size = 768
    model.scene_embedding_size = model.embedding_size
    model.timestamp_embedding_size = model.embedding_size

    return model


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
    if not isinstance(model, HuggingFaceWav2Vec2):
        raise ValueError(f"Model must be an instance of {HuggingFaceWav2Vec2.__name__}")

    # Send the model to the same device that the audio tensor is on.
    # model = model.to(audio.device)

    # Put the model into eval mode, and not computing gradients while in inference.
    # Iterate over all batches and accumulate the embeddings for each frame.
    model.eval()
    with torch.no_grad():
        embeddings = model(audio)

    # Length of the audio in MS
    audio_ms = int(audio.shape[1] / model.sample_rate * 1000)

    # samples => timestamps
    # 31439 => 97
    # 31440 => 98
    # This is weird that its 5ms, not half the hopsize of 20
    ntimestamps = (audio_ms - 5) // 20

    # Also
    # 32000 => 99
    # 32080 => 100

    # I don't know if this is their exact centering, but this matches
    # their shape.
    last_center = 12.5 + (ntimestamps - 1) * 20
    timestamps = torch.arange(12.5, last_center + 20, 20)
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
