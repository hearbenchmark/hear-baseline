"""
torchcrepe model for HEAR 2021 NeurIPS competition.
"""

from typing import Tuple

import torch
import torchcrepe
from torch import Tensor

SAMPLE_RATE = 16000

TIMESTAMP_HOP_SIZE = 50
SCENE_HOP_SIZE = 250

TIMESTAMP_HOP_SIZE_SAMPLES = (SAMPLE_RATE * TIMESTAMP_HOP_SIZE) // 1000
SCENE_HOP_SIZE_SAMPLES = (SAMPLE_RATE * SCENE_HOP_SIZE) // 1000


class TorchCrepeModel(torch.nn.Module):
    """
    A pretty gross wrapper on torchcrepe, because of its implicit singleton
    model loading: https://github.com/maxrmorrison/torchcrepe/issues/13
    """

    # sample rate and embedding sizes are required model attributes for the HEAR API
    sample_rate = SAMPLE_RATE
    # ???
    embedding_size = 32 * 64
    scene_embedding_size = embedding_size
    timestamp_embedding_size = embedding_size

    def __init__(self):
        super().__init__()

        # This is gross.
        if torch.cuda.is_available():
            torchcrepe.load.model(device="cuda", capacity="full")
        else:
            torchcrepe.load.model(device="cpu", capacity="full")

    def forward(self, x: Tensor, hop_size_samples: int):
        # Or do x.device?
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        if x.ndim == 1:
            x = x.view(1, x.shape[0])

        assert x.ndim == 2

        # This is lame, sorry
        # torchcrepe only can process one audio at a time
        embeddings = []
        for i in range(x.shape[0]):
            embedding = torchcrepe.embed(
                audio=x[i].view(1, x.shape[1]),
                sample_rate=self.sample_rate,
                hop_length=hop_size_samples,
                model="full",
                device=device,
                pad=True,
                # Otherwise dcase exceeds memory on a V100
                batch_size=512,
            )
            # Convert 1 x frames x 32x64 embedding to 1 x frames x 32*64
            assert embedding.shape[0] == 1
            assert embedding.ndim == 4
            embedding = embedding.view((1, embedding.shape[1], -1))
            embeddings.append(embedding)
        return torch.cat(embeddings)


def load_model(model_file_path: str = "") -> torch.nn.Module:
    """
    Returns a torch.nn.Module that produces embeddings for audio.

    Args:
        model_file_path: Ignored.
    Returns:
        Model
    """
    return TorchCrepeModel()


def get_timestamp_embeddings(
    audio: Tensor,
    model: torch.nn.Module,
    hop_size_samples: int = TIMESTAMP_HOP_SIZE_SAMPLES,
) -> Tuple[Tensor, Tensor]:
    """
    This function returns embeddings at regular intervals centered at timestamps. Both
    the embeddings and corresponding timestamps (in milliseconds) are returned.

    Args:
        audio: n_sounds x n_samples of mono audio in the range [-1, 1].
        model: Loaded model.
        hop_size_samples: Hop size in samples.

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
    if not isinstance(model, TorchCrepeModel):
        raise ValueError(f"Model must be an instance of {TorchCrepeModel.__name__}")

    # Send the model to the same device that the audio tensor is on.
    # model = model.to(audio.device)

    # Put the model into eval mode, and not computing gradients while in inference.
    # Iterate over all batches and accumulate the embeddings for each frame.
    model.eval()
    with torch.no_grad():
        embeddings = model(audio, hop_size_samples)

    ntimestamps = audio.shape[1] // hop_size_samples + 1
    hop_size = hop_size_samples * 1000 // SAMPLE_RATE

    # By default, the audio is padded with window_size // 2 zeros
    # on both sides. So a signal x will produce 1 + int(len(x) //
    # hop_size) frames. The first frame is centered on sample index
    # 0.
    # https://github.com/maxrmorrison/torchcrepe/issues/14
    timestamps = torch.tensor(
        [i * hop_size for i in range(ntimestamps)], device=embeddings.device
    )
    assert len(timestamps) == ntimestamps
    timestamps = timestamps.expand((embeddings.shape[0], timestamps.shape[0]))
    assert (
        timestamps.shape[1] == embeddings.shape[1]
    ), f"{timestamps.shape} vs {embeddings.shape}"

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
    embeddings, _ = get_timestamp_embeddings(
        audio, model, hop_size_samples=SCENE_HOP_SIZE_SAMPLES
    )
    embeddings = torch.mean(embeddings, dim=1)
    return embeddings
