"""
Tests for the TensorFlow baseline implementation
"""

import numpy as np
import torch
import tensorflow as tf

import hearbaseline.tf.naive as tf_baseline
import hearbaseline as torch_baseline


class TestTFNaiveModel:
    def test_model(self):

        model = tf_baseline.load_model()

        assert model.sample_rate == 44100
        assert model.scene_embedding_size == 4096
        assert model.timestamp_embedding_size == 4096

        assert hasattr(tf_baseline, "get_timestamp_embeddings")
        assert hasattr(tf_baseline, "get_scene_embeddings")

    def test_model_against_torch(self):

        model = tf_baseline.load_model()
        torch_model = torch_baseline.load_model()

        # Confirm that all the attributes are the same between tf and torch
        assert model.sample_rate == torch_model.sample_rate
        assert model.scene_embedding_size == torch_model.scene_embedding_size
        assert model.timestamp_embedding_size == torch_model.scene_embedding_size

        # Confirm that the projection matrix is the same shape
        assert model.projection.shape == torch_model.projection.shape


class TestTFNaiveEmbeddings:
    def setup(self):
        self.tf_model = tf_baseline.load_model()
        self.torch_model = torch_baseline.load_model()

        matrix_init = np.random.normal(size=self.tf_model.projection.shape)
        self.tf_model.projection.assign(
            tf.convert_to_tensor(matrix_init, dtype=tf.float32)
        )
        self.torch_model.projection.data = torch.tensor(
            matrix_init, dtype=torch.float32
        )

    def teardown(self):
        del self.tf_model
        del self.torch_model

    def test_scene_embeddings(self):

        # Confirm that all the projection matrix values are the same
        assert np.all(
            self.tf_model.projection.numpy()
            == self.torch_model.projection.detach().numpy()
        )

        num_audio = 4
        duration = 2.0
        torch_audio = (
            torch.rand((num_audio, int(self.torch_model.sample_rate * duration))) * 2
        ) - 1.0
        tf_audio = tf.convert_to_tensor(torch_audio.numpy())

        tf_embedding = tf_baseline.get_scene_embeddings(tf_audio, self.tf_model)
        torch_embedding = torch_baseline.get_scene_embeddings(
            torch_audio, self.torch_model
        )

        assert tf_embedding.shape == (num_audio, self.tf_model.scene_embedding_size)
        assert tf_embedding.shape == torch_embedding.shape

        error = np.mean(
            np.square(tf_embedding.numpy() - torch_embedding.detach().cpu().numpy())
        )
        assert error < 1e-7

    def test_timestamp_embeddings(self):

        # Confirm that all the projection matrix values are the same
        assert np.all(
            self.tf_model.projection.numpy()
            == self.torch_model.projection.detach().numpy()
        )

        num_audio = 4
        duration = 2.0
        torch_audio = (
            torch.rand((num_audio, int(self.torch_model.sample_rate * duration))) * 2
        ) - 1.0
        tf_audio = tf.convert_to_tensor(torch_audio.numpy())

        tf_embedding, tf_timestamps = tf_baseline.get_timestamp_embeddings(
            tf_audio, self.tf_model
        )
        torch_embedding, torch_timestamps = torch_baseline.get_timestamp_embeddings(
            torch_audio, self.torch_model
        )

        num_samples = duration * self.tf_model.sample_rate
        hop_size_samples = int(
            round(tf_baseline.HOP_SIZE / 1000.0 * self.tf_model.sample_rate)
        )
        expected_frames = int(num_samples // hop_size_samples + 1)

        assert tf_embedding.shape == (
            num_audio,
            expected_frames,
            self.tf_model.scene_embedding_size,
        )
        assert tf_embedding.shape == torch_embedding.shape

        # Confirm a small error between the embeddings
        error = np.mean(
            np.square(tf_embedding.numpy() - torch_embedding.detach().cpu().numpy())
        )
        assert error < 1e-7

        # Check timestamps
        assert np.allclose(
            tf_timestamps.numpy(), torch_timestamps.detach().cpu().numpy()
        )

    def test_timestamps_spacing(self):
        # Test the spacing between the time stamp
        audio = (tf.random.uniform((1, 96000)) * 2.0) - 1.0
        emb, ts = tf_baseline.get_timestamp_embeddings(audio, model=self.tf_model)
        timestamp_diff = ts[1:] - ts[:-1]
        assert tf.reduce_mean(timestamp_diff) - ts[1] < 1e-5
