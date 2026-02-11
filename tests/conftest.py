"""Shared pytest fixtures for GAVEL tests."""

import json
import tempfile
from pathlib import Path

import pytest
import torch


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_labels():
    """Create sample labels dictionary."""
    return {
        "making_a_threat": 0,
        "payment_processing": 1,
        "romance_baiting": 2,
        "credential_phishing": 3,
        "neutral_conversation": 4,
    }


@pytest.fixture
def sample_labels_file(temp_dir, sample_labels):
    """Create a temporary labels.json file."""
    labels_path = temp_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(sample_labels, f)
    return labels_path


@pytest.fixture
def sample_config_dict(temp_dir, sample_labels_file):
    """Create sample config dictionary."""
    return {
        "model_name": "test_model",
        "model": {
            "name_or_path": "mistralai/Mistral-7B-Instruct-v0.2",
            "selected_layers_range": [13, 27],
        },
        "training": {
            "batch_size": 32,
            "epochs": 10,
            "learning_rate": 3e-4,
        },
        "rnn": {
            "hidden_dim": 128,
            "num_rnn_layers": 2,
            "rnn_type": "GRU",
            "sequence_length": 5,
        },
        "paths": {
            "base_dir": str(temp_dir),
            "train_dataset": str(temp_dir / "train"),
            "eval_dataset": str(temp_dir / "eval"),
            "labels_path": str(sample_labels_file),
        },
        "eval": {
            "window_size": 5,
            "window_stride": 5,
        },
    }


@pytest.fixture
def sample_config_file(temp_dir, sample_config_dict):
    """Create a temporary config.json file."""
    config_path = temp_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(sample_config_dict, f)
    return config_path


@pytest.fixture
def mock_rnn_checkpoint(temp_dir):
    """Create a mock RNN model checkpoint."""
    from gavel.models.rnn import TopicRNN

    model = TopicRNN(
        input_dim=1024,
        num_layers=14,
        hidden_dim=128,
        num_rnn_layers=2,
        num_topics=5,
        rnn_type="GRU",
    )
    checkpoint_path = temp_dir / "model.pth"
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


@pytest.fixture
def sample_logits():
    """Create sample logits tensor for testing."""
    # Shape: (num_windows, num_labels)
    return torch.randn(10, 5)


@pytest.fixture
def sample_thresholds():
    """Create sample thresholds for testing."""
    return torch.tensor([0.5, 0.6, 0.7, 0.5, 0.4])
