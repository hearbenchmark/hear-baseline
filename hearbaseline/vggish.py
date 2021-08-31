"""
vggish model for HEAR 2021 NeurIPS competition.
"""

from typing import Tuple

import torch
import torchvggish.vggish_params
from torch import Tensor


class VggishWrapper(torch.nn.Module):
    # sample rate and embedding sizes are required model attributes for the HEAR API
    sample_rate = 16000
    embedding_size = 128
    scene_embedding_size = embedding_size
    timestamp_embedding_size = embedding_size

    def __init__(self):
        super().__init__()

        self.model = torch.hub.load("harritaylor/torchvggish", "vggish")
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    def forward(self, x: Tensor):
        # This is lame, sorry
        # vggish only can process one audio at a time
        embeddings = []
        for i in range(x.shape[0]):
            # tensor => numpy sucks too
            embedding = self.model(x[i].detach().cpu().numpy(), fs=self.sample_rate)
            # This is weird too
            if embedding.ndim == 1:
                assert embedding.shape[0] == 128
                embedding = embedding.view(1, 128)
            embeddings.append(embedding)
        embeddings = torch.stack(embeddings)
        return embeddings


def load_model(model_file_path: str = "", hop_length: int = 25) -> torch.nn.Module:
    """
    Returns a torch.nn.Module that produces embeddings for audio.

    Args:
        model_file_path: Ignored.
        hop_length: hop length in milliseconds. (Default: 25, even
        though the vggish default is 960, so we can do timestamp
        embeddings for event detection.)
    Returns:
        Model
    """
    torchvggish.vggish_params.EXAMPLE_HOP_SECONDS = hop_length / 1000

    return VggishWrapper()


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
    if not isinstance(model, VggishWrapper):
        raise ValueError(f"Model must be an instance of {VggishWrapper.__name__}")

    # Send the model to the same device that the audio tensor is on.
    # model = model.to(audio.device)

    # Put the model into eval mode, and not computing gradients while in inference.
    # Iterate over all batches and accumulate the embeddings for each frame.
    with torch.no_grad():
        embeddings = model(audio)

    # Length of the audio in MS
    # audio_ms = audio.shape[1] / model.sample_rate * 1000
    hop_length = int(torchvggish.vggish_params.EXAMPLE_HOP_SECONDS * 1000)
    # BUG: This sort of thing is likely to mess up the timestamps
    # since we don't understand precisely how they frame.
    # I don't know why this doesn't work
    # ntimestamps = int(audio_ms / hop_length)
    # so just hack it
    ntimestamps = embeddings.shape[1]

    last_center = int(hop_length * (ntimestamps + 0.5))
    timestamps = torch.arange(hop_length // 2, last_center, hop_length)
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
