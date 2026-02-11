"""Tests for gavel.preprocessing module."""

import torch

from gavel.preprocessing.utils import (
    _contiguous_true_runs,
    _kmp_find_all,
    _normalize_to_conversations,
    collate_fn,
)


class TestKMPFindAll:
    """Tests for KMP pattern matching utility."""

    def test_find_single_match(self):
        """Test finding a single pattern occurrence."""
        sequence = [1, 2, 3, 4, 5]
        pattern = [3, 4]
        matches = list(_kmp_find_all(sequence, pattern))
        assert matches == [2]

    def test_find_multiple_matches(self):
        """Test finding multiple non-overlapping matches."""
        sequence = [1, 2, 1, 2, 1, 2]
        pattern = [1, 2]
        matches = list(_kmp_find_all(sequence, pattern))
        assert matches == [0, 2, 4]

    def test_no_match(self):
        """Test when pattern is not found."""
        sequence = [1, 2, 3, 4, 5]
        pattern = [6, 7]
        matches = list(_kmp_find_all(sequence, pattern))
        assert matches == []

    def test_empty_pattern(self):
        """Test empty pattern returns nothing."""
        sequence = [1, 2, 3]
        pattern = []
        matches = list(_kmp_find_all(sequence, pattern))
        assert matches == []

    def test_pattern_longer_than_sequence(self):
        """Test pattern longer than sequence returns nothing."""
        sequence = [1, 2]
        pattern = [1, 2, 3, 4]
        matches = list(_kmp_find_all(sequence, pattern))
        assert matches == []


class TestContiguousTrueRuns:
    """Tests for extracting contiguous True runs from boolean masks."""

    def test_single_run(self):
        """Test single contiguous True span."""
        mask = torch.tensor([False, True, True, True, False])
        runs = _contiguous_true_runs(mask)
        assert runs == [(1, 4)]

    def test_multiple_runs(self):
        """Test multiple separate True spans."""
        mask = torch.tensor([True, True, False, True, False, True, True])
        runs = _contiguous_true_runs(mask)
        assert runs == [(0, 2), (3, 4), (5, 7)]

    def test_no_true_values(self):
        """Test mask with no True values."""
        mask = torch.tensor([False, False, False])
        runs = _contiguous_true_runs(mask)
        assert runs == []

    def test_all_true(self):
        """Test mask with all True values."""
        mask = torch.tensor([True, True, True])
        runs = _contiguous_true_runs(mask)
        assert runs == [(0, 3)]

    def test_empty_mask(self):
        """Test empty mask."""
        mask = torch.tensor([], dtype=torch.bool)
        runs = _contiguous_true_runs(mask)
        assert runs == []


class TestCollateFn:
    """Tests for batch collation function."""

    def test_collate_same_length(self):
        """Test collating batches of same length sequences."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "assistant_mask": torch.tensor([False, True, True]),
                "pad_token_id": 0,
            },
            {
                "input_ids": torch.tensor([4, 5, 6]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "assistant_mask": torch.tensor([True, True, False]),
                "pad_token_id": 0,
            },
        ]

        result = collate_fn(batch)

        assert result["input_ids"].shape == (2, 3)
        assert result["attention_mask"].shape == (2, 3)
        assert result["assistant_mask"].shape == (2, 3)

    def test_collate_different_lengths(self):
        """Test collating batches with different length sequences (padding)."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": torch.tensor([1, 1]),
                "assistant_mask": torch.tensor([False, True]),
                "pad_token_id": 0,
            },
            {
                "input_ids": torch.tensor([3, 4, 5, 6]),
                "attention_mask": torch.tensor([1, 1, 1, 1]),
                "assistant_mask": torch.tensor([True, True, True, True]),
                "pad_token_id": 0,
            },
        ]

        result = collate_fn(batch)

        # Should pad to max length (4)
        assert result["input_ids"].shape == (2, 4)
        # First sequence should be padded with 0
        assert result["input_ids"][0, 2].item() == 0
        assert result["input_ids"][0, 3].item() == 0
        # Padding positions should have attention_mask = 0
        assert result["attention_mask"][0, 2].item() == 0
        assert result["attention_mask"][0, 3].item() == 0

    def test_collate_preserves_assistant_mask(self):
        """Test that assistant mask is correctly preserved after collation."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "assistant_mask": torch.tensor([False, True, True]),
                "pad_token_id": 0,
            },
        ]

        result = collate_fn(batch)

        assert result["assistant_mask"][0, 0].item() == False
        assert result["assistant_mask"][0, 1].item() == True
        assert result["assistant_mask"][0, 2].item() == True


class TestNormalizeToConversations:
    """Tests for conversation normalization utility."""

    def test_normalize_chat_format(self):
        """Test normalizing standard chat format."""
        data = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        convs = _normalize_to_conversations(data)

        assert len(convs) == 1
        assert len(convs[0]) == 2
        assert convs[0][0]["role"] == "user"
        assert convs[0][1]["role"] == "assistant"

    def test_normalize_missing_role_defaults_to_user(self):
        """Test that missing role defaults to 'user'."""
        data = [
            {"content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        convs = _normalize_to_conversations(data)

        assert len(convs) == 1
        # Without role, should treat as individual user messages
        # but since one has role, it's treated as conversation
        assert convs[0][0]["role"] == "user"

    def test_normalize_empty_content_skipped(self):
        """Test that empty content messages are skipped."""
        data = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},  # Empty
            {"role": "assistant", "content": "Valid response"},
        ]

        convs = _normalize_to_conversations(data)

        # Empty content should be filtered
        assert len(convs) == 1
        assert len(convs[0]) == 2  # Only 2 valid messages

    def test_normalize_legacy_statement_format(self):
        """Test normalizing legacy 'Statement' format."""
        data = [
            {"Statement": "Legacy user input"},
        ]

        convs = _normalize_to_conversations(data)

        # Should handle Statement fallback
        assert len(convs) == 1
        assert convs[0][0]["content"] == "Legacy user input"

    def test_normalize_invalid_role_defaults_to_user(self):
        """Test that invalid roles default to 'user'."""
        data = [
            {"role": "invalid_role", "content": "Test"},
        ]

        convs = _normalize_to_conversations(data)

        assert convs[0][0]["role"] == "user"
