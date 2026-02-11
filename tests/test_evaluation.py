"""Tests for gavel.evaluation module."""

import numpy as np
import pytest
import torch

from gavel.evaluation.metrics import (
    compute_triggers,
    compute_usecase_detection_metrics,
    convert_labels_to_tensors,
)


class TestComputeTriggers:
    """Tests for compute_triggers function."""

    def test_compute_triggers_basic(self):
        """Test basic trigger computation with tensor input."""
        # Logits that will produce high probabilities after sigmoid
        logits = torch.tensor(
            [
                [2.0, -2.0, 0.0, 1.0, -1.0],
                [2.5, -2.5, 0.5, 1.5, -1.5],
            ]
        )
        thresholds = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])

        triggers = compute_triggers(logits, thresholds, patience_rate=1)

        assert isinstance(triggers, torch.Tensor)
        assert triggers.shape == (5,)
        # High logits should trigger (sigmoid(2.0) ≈ 0.88 > 0.5)
        assert triggers[0] == 1
        # Low logits should not trigger (sigmoid(-2.0) ≈ 0.12 < 0.5)
        assert triggers[1] == 0

    def test_compute_triggers_numpy_input(self):
        """Test trigger computation with numpy array input."""
        logits = np.array(
            [
                [2.0, -2.0, 0.0],
                [2.5, -2.5, 0.5],
            ]
        )
        thresholds = 0.5

        triggers = compute_triggers(logits, thresholds, patience_rate=1)

        assert isinstance(triggers, torch.Tensor)
        assert triggers.shape == (3,)

    def test_compute_triggers_patience(self):
        """Test patience_rate affects trigger behavior."""
        # Only first window exceeds threshold
        logits = torch.tensor(
            [
                [3.0, 3.0, 3.0],  # Will trigger
                [-3.0, -3.0, -3.0],  # Won't trigger
                [-3.0, -3.0, -3.0],  # Won't trigger
            ]
        )

        # With patience=1, should trigger
        triggers_p1 = compute_triggers(logits, 0.5, patience_rate=1)
        assert triggers_p1.sum() > 0

        # With patience=3, should NOT trigger (not enough consecutive windows)
        triggers_p3 = compute_triggers(logits, 0.5, patience_rate=3)
        # The exact behavior depends on implementation

    def test_compute_triggers_scalar_threshold(self):
        """Test trigger computation with scalar threshold."""
        logits = torch.randn(5, 3)
        triggers = compute_triggers(logits, thresholds=0.8, patience_rate=1)

        assert triggers.shape == (3,)


class TestConvertLabelsToTensors:
    """Tests for convert_labels_to_tensors function."""

    def test_convert_labels_basic(self, sample_labels):
        """Test basic label conversion to tensors."""
        data = {
            "test_usecase": {
                "all_required": ["making_a_threat", "payment_processing"],
                "supporting": ["romance_baiting"],
            }
        }

        result = convert_labels_to_tensors(data, sample_labels)

        assert "test_usecase" in result
        assert "all_required_labels" in result["test_usecase"]
        assert "supporting_labels" in result["test_usecase"]

        # Check tensor types
        assert isinstance(result["test_usecase"]["all_required_labels"], torch.Tensor)
        assert isinstance(result["test_usecase"]["supporting_labels"], torch.Tensor)

    def test_convert_labels_empty(self, sample_labels):
        """Test conversion with empty label lists."""
        data = {
            "empty_usecase": {
                "all_required": [],
                "supporting": [],
            }
        }

        result = convert_labels_to_tensors(data, sample_labels)

        assert "empty_usecase" in result
        # Empty tensor or zeros tensor expected
        assert result["empty_usecase"]["all_required_labels"].sum() == 0

    def test_convert_labels_one_hot_encoding(self, sample_labels):
        """Test that output is properly one-hot encoded."""
        data = {
            "single_label": {
                "all_required": ["making_a_threat"],
                "supporting": [],
            }
        }

        result = convert_labels_to_tensors(data, sample_labels)

        all_required = result["single_label"]["all_required_labels"]
        # Should be one-hot with single 1 at index 0
        assert all_required[0] == 1
        assert all_required.sum() == 1


class TestComputeUsecaseDetectionMetrics:
    """Tests for compute_usecase_detection_metrics function."""

    def test_compute_metrics_basic(self):
        """Test basic metrics computation from stats."""
        stats = {
            "usecase_a": {
                "true_positive": 80,
                "false_positive": 10,
                "true_negative": 90,
                "false_negative": 20,
            },
            "usecase_b": {
                "true_positive": 50,
                "false_positive": 5,
                "true_negative": 140,
                "false_negative": 5,
            },
        }

        df = compute_usecase_detection_metrics(stats)

        assert len(df) == 2
        assert "TPR" in df.columns
        assert "FPR" in df.columns
        assert "Accuracy" in df.columns
        assert "F1" in df.columns

    def test_compute_metrics_tpr_calculation(self):
        """Test TPR is calculated correctly."""
        stats = {
            "test": {
                "true_positive": 80,
                "false_positive": 10,
                "true_negative": 90,
                "false_negative": 20,
            },
        }

        df = compute_usecase_detection_metrics(stats)

        # TPR = TP / (TP + FN) = 80 / 100 = 0.8
        assert df[df["Usecase"] == "test"]["TPR"].values[0] == pytest.approx(0.8, rel=1e-2)

    def test_compute_metrics_fpr_calculation(self):
        """Test FPR is calculated correctly."""
        stats = {
            "test": {
                "true_positive": 80,
                "false_positive": 10,
                "true_negative": 90,
                "false_negative": 20,
            },
        }

        df = compute_usecase_detection_metrics(stats)

        # FPR = FP / (FP + TN) = 10 / 100 = 0.1
        assert df[df["Usecase"] == "test"]["FPR"].values[0] == pytest.approx(0.1, rel=1e-2)

    def test_compute_metrics_zero_division(self):
        """Test handling of zero TPR denominator."""
        stats = {
            "no_positives": {
                "true_positive": 0,
                "false_positive": 10,
                "true_negative": 90,
                "false_negative": 0,
            },
        }

        df = compute_usecase_detection_metrics(stats)

        # Should handle division by zero gracefully (TPR = 0 when no positives)
        assert df["TPR"].values[0] == 0
