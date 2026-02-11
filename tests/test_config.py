"""Tests for gavel.config module."""

import json

import pytest

from gavel.config import (
    GavelConfig,
    ModelConfig,
    PathsConfig,
    RNNConfig,
    TrainingConfig,
    load_config,
    load_labels,
)


class TestLoadLabels:
    """Tests for load_labels function."""

    def test_load_labels_valid(self, sample_labels_file, sample_labels):
        """Test loading valid labels file."""
        labels = load_labels(str(sample_labels_file))
        assert labels == sample_labels
        assert len(labels) == 5

    def test_load_labels_missing_file(self, temp_dir):
        """Test FileNotFoundError for missing labels file."""
        with pytest.raises(FileNotFoundError):
            load_labels(str(temp_dir / "nonexistent.json"))

    def test_load_labels_with_base_dir(self, temp_dir, sample_labels):
        """Test loading labels with base_dir resolution."""
        labels_path = temp_dir / "subdir" / "labels.json"
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        with open(labels_path, "w") as f:
            json.dump(sample_labels, f)

        labels = load_labels("subdir/labels.json", base_dir=str(temp_dir))
        assert labels == sample_labels


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_valid(self, sample_config_file, sample_labels):
        """Test loading valid config file."""
        config = load_config(str(sample_config_file))

        assert isinstance(config, GavelConfig)
        assert config.model_name == "test_model"
        assert config.labels == sample_labels
        assert config.num_labels == 5

    def test_load_config_missing_file(self, temp_dir):
        """Test FileNotFoundError for missing config file."""
        with pytest.raises(FileNotFoundError):
            load_config(str(temp_dir / "nonexistent.json"))

    def test_load_config_model_settings(self, sample_config_file):
        """Test model configuration is parsed correctly."""
        config = load_config(str(sample_config_file))

        assert config.model.name_or_path == "mistralai/Mistral-7B-Instruct-v0.2"
        assert config.model.selected_layers_range == (13, 27)
        assert list(config.model.selected_layers) == list(range(13, 27))

    def test_load_config_training_settings(self, sample_config_file):
        """Test training configuration is parsed correctly."""
        config = load_config(str(sample_config_file))

        assert config.training.batch_size == 32
        assert config.training.epochs == 10
        assert config.training.learning_rate == 3e-4

    def test_load_config_rnn_settings(self, sample_config_file):
        """Test RNN configuration is parsed correctly."""
        config = load_config(str(sample_config_file))

        assert config.rnn.hidden_dim == 128
        assert config.rnn.num_rnn_layers == 2
        assert config.rnn.rnn_type == "GRU"
        assert config.rnn.sequence_length == 5


class TestConfigDataclasses:
    """Tests for configuration dataclasses."""

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()
        assert config.name_or_path == "mistralai/Mistral-7B-Instruct-v0.2"
        assert config.selected_layers_range == (13, 27)

    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()
        assert config.batch_size == 64
        assert config.epochs == 25
        assert config.early_stopping is True

    def test_rnn_config_defaults(self):
        """Test RNNConfig default values."""
        config = RNNConfig()
        assert config.hidden_dim == 256
        assert config.rnn_type == "GRU"
        assert config.proj_dim is None

    def test_paths_config_derived_paths(self, temp_dir):
        """Test PathsConfig derived path properties."""
        config = PathsConfig(base_dir=str(temp_dir))

        assert config.logits_dir == str(temp_dir / "logits")
        assert config.model_dir == str(temp_dir / "model")
        assert config.rnn_model_path == str(temp_dir / "model" / "trained_model_rnn.pth")
        assert config.calibration_dir == str(temp_dir / "calibration_results")

    def test_gavel_config_index_to_label(self, sample_labels):
        """Test GavelConfig reverse label mapping."""
        config = GavelConfig(labels=sample_labels)

        idx_to_label = config.index_to_label
        assert idx_to_label[0] == "making_a_threat"
        assert idx_to_label[4] == "neutral_conversation"
