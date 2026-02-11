"""
GAVEL: Attention-based Classification for Dialogue Analysis

A package for training and evaluating attention-based RNN classifiers
on dialogue data for detecting sensitive topics.
"""

__version__ = "0.1.0"

# Config utilities
from gavel.config import load_config, load_labels, GavelConfig

# Model classes and functions
from gavel.models.rnn import TopicRNN, train_rnn_model, load_trained_classifier

# Evaluation functions
from gavel.evaluation import evaluate, calibrate

__all__ = [
    # Version
    "__version__",
    # Config
    "load_config",
    "load_labels",
    "GavelConfig",
    # Models
    "TopicRNN",
    "train_rnn_model",
    "load_trained_classifier",
    # Evaluation
    "evaluate",
    "calibrate",
]

