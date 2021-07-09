"""
Utility functions for hear-kit
"""

from typing import Tuple

import tensorflow as tf


def frame_audio(
    audio: tf.Tensor, frame_size: int, hop_size: float, sample_rate: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Slices input audio into frames that are centered and occur every
    sample_rate * hop_size samples. We round to the nearest sample.

    Args:
        audio: input audio, expects a 2d Tensor of shape:
            (batch_size, num_samples)
        frame_size: the number of samples each resulting frame should be
        hop_size: hop size between frames, in milliseconds
        sample_rate: sampling rate of the input audio

    Returns:
        - A Tensor of shape (batch_size, num_frames, frame_size)
        - A 1d Tensor of timestamps corresponding to the frame
        centers.
    """

    # Zero pad the beginning and the end of the incoming audio with half a frame number
    # of samples. This centers the audio in the middle of each frame with respect to
    # the timestamps.
    paddings = tf.constant([[0, 0], [frame_size // 2, frame_size - frame_size // 2]])
    audio = tf.pad(audio, paddings)

    # Split into frames
    frame_step = int(hop_size / 1000.0 * sample_rate)
    frames = tf.signal.frame(audio, frame_length=frame_size, frame_step=frame_step)

    # Timestamps in ms corresponding to each frame
    num_frames = frames.shape[1]
    timestamps = tf.range(0, num_frames, dtype=tf.float32) * frame_step
    timestamps = timestamps / sample_rate * 1000.0

    return frames, timestamps
