"""Tests for gavel.evaluation.calibration module."""

import pandas as pd
import pytest
import torch

from gavel.evaluation.calibration import (
    compute_optimal_params,
    update_label_level_stats,
)



class TestComputeOptimalParams:
    """Tests for Youden's J-statistic optimization."""

    def test_youden_j_calculation(self):
        """Test that Youden's J is calculated as TPR - FPR."""
        df = pd.DataFrame(
            {
                "Topic": ["topic_a", "topic_a", "topic_a"],
                "Patience": [1, 1, 1],
                "Threshold": [0.3, 0.5, 0.7],
                "TPR": [0.9, 0.8, 0.6],
                "FPR": [0.3, 0.1, 0.05],
            }
        )

        optimal, df_out = compute_optimal_params(df, "Topic", verbose=False)

        # Youden's J should be added
        assert "YoudenJ" in df_out.columns
        # J = TPR - FPR
        expected_j = [0.6, 0.7, 0.55]  # 0.9-0.3, 0.8-0.1, 0.6-0.05
        assert list(df_out["YoudenJ"]) == pytest.approx(expected_j, rel=1e-2)

    def test_optimal_threshold_selection(self):
        """Test that optimal threshold maximizes Youden's J."""
        df = pd.DataFrame(
            {
                "Topic": ["topic_a", "topic_a", "topic_a"],
                "Patience": [1, 1, 1],
                "Threshold": [0.3, 0.5, 0.7],
                "TPR": [0.9, 0.8, 0.6],
                "FPR": [0.3, 0.1, 0.05],
            }
        )

        optimal, _ = compute_optimal_params(df, "Topic", verbose=False)

        # Best J is at threshold 0.5 (J=0.7)
        assert optimal["topic_a"]["threshold"] == 0.5
        assert optimal["topic_a"]["youden_j"] == pytest.approx(0.7, rel=1e-2)

    def test_multiple_topics(self):
        """Test optimization across multiple topics."""
        df = pd.DataFrame(
            {
                "Topic": ["topic_a", "topic_a", "topic_b", "topic_b"],
                "Patience": [1, 1, 1, 1],
                "Threshold": [0.3, 0.7, 0.4, 0.8],
                "TPR": [0.9, 0.6, 0.85, 0.5],
                "FPR": [0.2, 0.05, 0.15, 0.02],
            }
        )

        optimal, _ = compute_optimal_params(df, "Topic", verbose=False)

        assert "topic_a" in optimal
        assert "topic_b" in optimal

    def test_tie_breaking_prefers_higher_threshold(self):
        """Test that ties in J prefer higher threshold (more conservative)."""
        df = pd.DataFrame(
            {
                "Topic": ["topic_a", "topic_a"],
                "Patience": [1, 1],
                "Threshold": [0.3, 0.7],
                "TPR": [0.8, 0.7],
                "FPR": [0.1, 0.0],  # Both have J = 0.7
            }
        )

        optimal, _ = compute_optimal_params(df, "Topic", verbose=False)

        # Should prefer higher threshold when J values are within tolerance
        assert optimal["topic_a"]["threshold"] == 0.7

    def test_returns_tpr_fpr_at_optimal(self):
        """Test that optimal params include TPR/FPR at the selected threshold."""
        df = pd.DataFrame(
            {
                "Topic": ["topic_a", "topic_a"],
                "Patience": [1, 1],
                "Threshold": [0.5, 0.8],
                "TPR": [0.9, 0.7],
                "FPR": [0.2, 0.1],
            }
        )

        optimal, _ = compute_optimal_params(df, "Topic", verbose=False)

        # J at 0.5 = 0.7, J at 0.8 = 0.6 → optimal is 0.5
        assert optimal["topic_a"]["tpr_at_optimal"] == 0.9
        assert optimal["topic_a"]["fpr_at_optimal"] == 0.2


class TestUpdateLabelLevelStats:
    """Tests for rule-based label statistics update."""

    @pytest.fixture
    def empty_stats(self):
        """Create empty stats for 5 labels."""
        return [
            {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
            for _ in range(5)
        ]

    def test_all_required_triggered_is_tp(self, empty_stats):
        """Test that triggering all required labels counts as TP."""
        triggers = torch.tensor([1, 1, 0, 0, 0], dtype=torch.float32)
        all_required = torch.tensor([1, 1, 0, 0, 0], dtype=torch.float32)
        supporting = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
        any_of_conditions = {}

        update_label_level_stats(
            triggers, all_required, supporting, any_of_conditions, "test_uc", empty_stats
        )

        # Indices 0, 1 were required and triggered → TP
        assert empty_stats[0]["true_positive"] == 1
        assert empty_stats[1]["true_positive"] == 1

    def test_all_required_missed_is_fn(self, empty_stats):
        """Test that missing all required labels counts as FN."""
        triggers = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
        all_required = torch.tensor([1, 1, 0, 0, 0], dtype=torch.float32)
        supporting = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
        any_of_conditions = {}

        update_label_level_stats(
            triggers, all_required, supporting, any_of_conditions, "test_uc", empty_stats
        )

        # Indices 0, 1 were required but not triggered → FN
        assert empty_stats[0]["false_negative"] == 1
        assert empty_stats[1]["false_negative"] == 1

    def test_supporting_triggered_is_tp(self, empty_stats):
        """Test that triggering supporting labels counts as TP."""
        triggers = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32)
        all_required = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
        supporting = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32)
        any_of_conditions = {}

        update_label_level_stats(
            triggers, all_required, supporting, any_of_conditions, "test_uc", empty_stats
        )

        # Index 2 is supporting and was triggered → TP
        assert empty_stats[2]["true_positive"] == 1

    def test_supporting_not_triggered_is_tn(self, empty_stats):
        """Test that NOT triggering supporting labels counts as TN."""
        triggers = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
        all_required = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
        supporting = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32)
        any_of_conditions = {}

        update_label_level_stats(
            triggers, all_required, supporting, any_of_conditions, "test_uc", empty_stats
        )

        # Index 2 is supporting but not triggered → TN
        assert empty_stats[2]["true_negative"] == 1

    def test_irrelevant_triggered_is_fp(self, empty_stats):
        """Test that triggering irrelevant labels counts as FP."""
        triggers = torch.tensor([0, 0, 0, 0, 1], dtype=torch.float32)
        all_required = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)  # Only 0 is required
        supporting = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
        any_of_conditions = {}

        update_label_level_stats(
            triggers, all_required, supporting, any_of_conditions, "test_uc", empty_stats
        )

        # Index 4 is irrelevant (not required/supporting) but triggered → FP
        assert empty_stats[4]["false_positive"] == 1

    def test_irrelevant_not_triggered_is_tn(self, empty_stats):
        """Test that NOT triggering irrelevant labels counts as TN."""
        triggers = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)
        all_required = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)
        supporting = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
        any_of_conditions = {}

        update_label_level_stats(
            triggers, all_required, supporting, any_of_conditions, "test_uc", empty_stats
        )

        # Indices 1-4 are irrelevant and not triggered → TN
        assert empty_stats[1]["true_negative"] == 1
        assert empty_stats[4]["true_negative"] == 1

    def test_any_of_one_triggered_is_tp(self, empty_stats):
        """Test any-of logic: triggering at least one from group is TP."""
        triggers = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32)
        all_required = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
        supporting = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
        any_of_conditions = {
            "test_uc": [torch.tensor([2, 3])]  # Either 2 or 3 should trigger
        }

        update_label_level_stats(
            triggers, all_required, supporting, any_of_conditions, "test_uc", empty_stats
        )

        # Index 2 triggered (from any_of group) → TP
        assert empty_stats[2]["true_positive"] == 1
        # Index 3 not triggered but one from group did → TN
        assert empty_stats[3]["true_negative"] == 1

    def test_any_of_none_triggered_is_fn(self, empty_stats):
        """Test any-of logic: triggering none from group is FN for all in group."""
        triggers = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
        all_required = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
        supporting = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
        any_of_conditions = {
            "test_uc": [torch.tensor([2, 3])]  # Either 2 or 3 should trigger
        }

        update_label_level_stats(
            triggers, all_required, supporting, any_of_conditions, "test_uc", empty_stats
        )

        # Neither 2 nor 3 triggered → FN for both
        assert empty_stats[2]["false_negative"] == 1
        assert empty_stats[3]["false_negative"] == 1

    def test_error_on_overlapping_required_and_supporting(self, empty_stats):
        """Test that overlapping required/supporting raises error."""
        triggers = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)
        all_required = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)
        supporting = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)  # Overlap!
        any_of_conditions = {}

        with pytest.raises(ValueError, match="both all_required and supporting"):
            update_label_level_stats(
                triggers, all_required, supporting, any_of_conditions, "test_uc", empty_stats
            )
