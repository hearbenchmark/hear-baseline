"""
TensorFlow2 Version of naive Baseline model for HEAR 2021 NeurIPS competition.

This is simply a mel spectrogram followed by random projection.
"""

from typing import Tuple

import librosa
import numpy as np
import tensorflow as tf

from hearbaseline.tf.util import frame_audio

# Default hop_size in milliseconds
TIMESTAMP_HOP_SIZE = 50
SCENE_HOP_SIZE = 250

# Number of frames to batch process for timestamp embeddings
BATCH_SIZE = 512


class RandomProjectionMelEmbedding(tf.Module):
    """
    Baseline audio embedding model. This model creates mel frequency spectrums with
    256 mel-bands, and then performs a projection to an embedding size of 4096.
    """

    # sample rate and embedding sizes are required model attributes for the HEAR API
    sample_rate = 44100
    embedding_size = 4096
    scene_embedding_size = embedding_size
    timestamp_embedding_size = embedding_size

    # These attributes are specific to this baseline model
    n_fft = 4096
    n_mels = 256
    seed = 0
    epsilon = 1e-4

    def __init__(self, name=None):
        super().__init__(name=name)

        # Create a Hann window buffer to apply to frames prior to FFT.
        self.window = tf.Variable(tf.signal.hann_window(self.n_fft), trainable=False)

        # Create a mel filter weight matrix.
        mel_scale: tf.Tensor = tf.convert_to_tensor(
            librosa.filters.mel(self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels)
        )
        self.mel_scale = tf.Variable(mel_scale, trainable=False)

        # Projection matrix.
        self.projection = tf.Variable(
            tf.random.uniform((self.n_mels, self.embedding_size))
        )

    def __call__(self, x: tf.Tensor):
        x = tf.signal.rfft(x * self.window)

        # Convert to a power spectrum
        x = tf.abs(x) ** 2.0

        # Apply the mel-scale filter to the power spectrum.
        x = tf.matmul(x, tf.transpose(self.mel_scale))

        # Convert to a log mel spectrum.
        x = tf.math.log(x + self.epsilon)

        # Apply projection to get a 4096 dimension embedding
        x = tf.matmul(x, self.projection)

        return x


def load_model(model_file_path: str = "") -> tf.Module:
    """
    Returns a tf.Module that produces embeddings for audio.

    Args:
        model_file_path: Load model checkpoint from this file path. For this baseline,
            if no path is provided then the default random init weights for the
            linear projection layer will be used.
    Returns:
        Model: tf.Module
    """
    model = RandomProjectionMelEmbedding()

    if model_file_path != "":
        # Since there are just projection weights for this naive baseline,
        # they've been saved as a npy file. For a more complicated tf Module
        # we could have used the tensorflow SavedModel format:
        # https://www.tensorflow.org/api_docs/python/tf/saved_model/load
        model_weights = np.load(model_file_path)
        assert model_weights.shape == model.projection.shape
        model.projection.assign(tf.convert_to_tensor(model_weights, dtype=tf.float32))
    else:
        # Randomly initialize weights from normal distribution
        rng = tf.random.Generator.from_seed(model.seed)
        weights = rng.normal(model.projection.shape)
        model.projection.assign(weights)

    return model


def get_timestamp_embeddings(
    audio: tf.Tensor,
    model: tf.Module,
    hop_size: float = TIMESTAMP_HOP_SIZE,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    This function returns embeddings at regular intervals centered at timestamps. Both
    the embeddings and corresponding timestamps (in milliseconds) are returned.

    Args:
        audio: n_sounds x n_samples of mono audio in the range [-1, 1].
        model: Loaded model.
        hop_size: Hop size in milliseconds.
            NOTE: Not required by the HEAR API. We add this optional parameter
            to improve the efficiency of scene embedding.

    Returns:
        - Tensor: embeddings, A float32 Tensor with shape (n_sounds, n_timestamp,
            model.timestamp_embedding_size).
        - Tensor: timestamps, Centered timestamps in milliseconds corresponding
            to each embedding in the output.
    """

    # Assert audio is of correct shape
    if audio.ndim != 2:
        raise ValueError(
            "audio input tensor must be 2D with shape (n_sounds, num_samples)"
        )

    # Make sure the correct model type was passed in
    if not isinstance(model, RandomProjectionMelEmbedding):
        raise ValueError(
            f"Model must be an instance of {RandomProjectionMelEmbedding.__name__}"
        )

    frames, timestamps = frame_audio(
        audio, frame_size=model.n_fft, hop_size=hop_size, sample_rate=model.sample_rate
    )

    # Combine all the frames from all audio batches together for batch processing
    # of frames. We'll unflatten these after processing through the model
    audio_batches, num_frames, frame_size = frames.shape
    frames = tf.reshape(frames, [audio_batches * num_frames, frame_size])

    embeddings_list = []
    for i in range(0, frames.shape[0], BATCH_SIZE):
        frame_batch = frames[i : i + BATCH_SIZE]
        embeddings_list.extend(model(frame_batch))

    # Unflatten all the frames back into audio batches
    embeddings = tf.stack(embeddings_list, axis=0)
    embeddings = tf.reshape(embeddings, (audio_batches, num_frames, frame_size))

    return embeddings, timestamps


def get_scene_embeddings(
    audio: tf.Tensor,
    model: tf.Module,
) -> tf.Tensor:
    """
    This function returns a single embedding for each audio clip. In this baseline
    implementation we simply summarize the temporal embeddings from
    get_timestamp_embeddings() using tf.reduce_mean().

    Args:
        audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in
            a batch will be padded/trimmed to the same length.
        model: Loaded model.

    Returns:
        - embeddings, A float32 Tensor with shape
            (n_sounds, model.scene_embedding_size).
    """
    embeddings, _ = get_timestamp_embeddings(audio, model, hop_size=SCENE_HOP_SIZE)
    embeddings = tf.reduce_mean(embeddings, axis=1)
    return embeddings
