"""
Utility functions for hear-kit
"""

from typing import Tuple

import tensorflow as tf


def frame_audio(audio: tf.Tensor, frame_size: int, hop_size: float,
                sample_rate: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Slices input audio into frames that are centered and occur every
    sample_rate * hop_size samples. We round to the nearest sample.

    Args:
        audio: input audio, expects a 2d Tensor of shape:
            (n_sounds, num_samples)
        frame_size: the number of samples each resulting frame should be
        hop_size: hop size between frames, in milliseconds
        sample_rate: sampling rate of the input audio

    Returns:
        - A Tensor of shape (n_sounds, num_frames, frame_size)
        - A Tensor of timestamps corresponding to the frame centers with shape:
            (n_sounds, num_frames).
    """

    # Zero pad the beginning and the end of the incoming audio with half a frame number
    # of samples. This centers the audio in the middle of each frame with respect to
    # the timestamps.
    paddings = tf.constant([[0, 0],
                            [frame_size // 2, frame_size - frame_size // 2]])
    audio = tf.pad(audio, paddings)
    num_padded_samples = audio.shape[1]

    # This implementation is really elegant but frame_step rounding errors
    # can accumulate into drift.
    """
    # Split into frames
    frame_step = int(round(hop_size / 1000.0 * sample_rate))
    frames = tf.signal.frame(audio, frame_length=frame_size, frame_step=frame_step)

    # Timestamps in ms corresponding to each frame
    num_frames = frames.shape[1]
    timestamps = tf.range(0, num_frames, dtype=tf.float32) * frame_step
    timestamps = timestamps / sample_rate * 1000.0

    # Expand out timestamps to shape (n_sounds, num_frames)
    timestamps = tf.expand_dims(timestamps, axis=0)
    timestamps = tf.repeat(timestamps, repeats=[audio.shape[0]], axis=0)
    """

    frame_step = hop_size / 1000.0 * sample_rate
    frame_number = 0
    frames = []
    timestamps = []
    frame_start = 0
    frame_end = frame_size
    while True:
        frames.append(audio[:, frame_start:frame_end])
        timestamps.append(frame_number * frame_step / sample_rate * 1000.0)

        # Increment the frame_number and break the loop if the next frame end
        # will extend past the end of the padded audio samples
        frame_number += 1
        frame_start = int(round(frame_number * frame_step))
        frame_end = frame_start + frame_size

        if not frame_end <= num_padded_samples:
            break

    # Expand out the timestamps to shape (n_sounds, num_frames)
    timestamps = tf.convert_to_tensor(timestamps)
    timestamps = tf.expand_dims(timestamps, axis=0)
    timestamps = tf.repeat(timestamps, repeats=[audio.shape[0]], axis=0)

    return tf.stack(frames, axis=1), timestamps
