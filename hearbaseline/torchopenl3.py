"""
torchopenl3 model for HEAR 2021 NeurIPS competition.
"""

import functools
from typing import Tuple

import torch
import torchopenl3
from torch import Tensor


def load_model(
    model_file_path: str = "",
    input_repr="mel256",
    content_type="music",
    embedding_size=6144,
    center=True,
    hop_size=0.05,
    batch_size=32,
    verbose=False,
) -> torch.nn.Module:
    """
    Returns a torch.nn.Module that produces embeddings for audio.

    Args:
        model_file_path: Ignored.
    Returns:
        Model
    """
    model = torchopenl3.core.load_audio_embedding_model(
        input_repr, content_type, embedding_size
    )
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    model.sample_rate = 48000
    model.embedding_size = embedding_size
    model.scene_embedding_size = embedding_size
    model.timestamp_embedding_size = embedding_size
    # model.center=center
    model.hop_size = hop_size
    # model.batch_size=batch_size
    # model.verbose=verbose

    model.get_audio_embedding = functools.partial(
        torchopenl3.core.get_audio_embedding,
        sr=model.sample_rate,
        model=model,
        center=center,
        hop_size=hop_size,
        batch_size=batch_size,
        verbose=verbose,
    )
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
    if not isinstance(model, torchopenl3.models.PytorchOpenl3):
        raise ValueError(
            f"Model must be an instance of {torchopenl3.models.PytorchOpenl3.__name__}"
        )

    # Send the model to the same device that the audio tensor is on.
    # model = model.to(audio.device)

    # Put the model into eval mode, and not computing gradients while in inference.
    # Iterate over all batches and accumulate the embeddings for each frame.
    with torch.no_grad():
        # Pad by up to 1/2 frame (0.5 seconds), so that we get a
        # timestamp at the end of audio

        padded_audio = torch.nn.functional.pad(
            audio,
            (
                0,
                int(model.sample_rate / 2)
                - audio.shape[1] % int(model.sample_rate * model.hop_size),
            ),
            mode="constant",
            value=0,
        )
        embeddings, timestamps = model.get_audio_embedding(padded_audio)

    # seconds to ms
    timestamps = timestamps * 1000
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
