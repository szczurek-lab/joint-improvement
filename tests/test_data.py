"""Tests for data loading utilities, including dataset and dataloader."""

from pathlib import Path

import pytest

from joint_improvement.utils.data.dataset import SequenceDataset, _resolve_data_path


def test_sequence_dataset_basic_access():
    """Test basic dataset access."""
    sequences = ["ACD", "EFG"]
    targets = [1, 2]

    dataset = SequenceDataset(sequences=sequences, targets=targets)

    assert len(dataset) == 2
    sample = dataset[1]
    assert sample[SequenceDataset.SEQUENCE_FIELD] == "EFG"
    assert sample[SequenceDataset.TARGET_FIELD] == 2


def test_sequence_dataset_applies_transforms():
    """Test that transforms are applied correctly."""
    def fake_tokenizer(text: str) -> str:
        return text.upper()

    sequences = ["hi"]
    targets = [0]
    target_transform = lambda value: value + 1
    
    dataset = SequenceDataset(
        sequences=sequences,
        targets=targets,
        transforms=[fake_tokenizer],
        target_transforms=[target_transform],
    )

    sample = dataset[0]
    assert sample[SequenceDataset.SEQUENCE_FIELD] == "HI"
    assert sample[SequenceDataset.TARGET_FIELD] == 1


def test_resolve_data_path_basic_usage():
    """Test basic usage with seed and split."""
    data_path = "seed_{seed}/{split}.npz"
    result = _resolve_data_path(data_path, split="train", seed=1337)
    assert result == Path("seed_1337/train.npz")


def test_resolve_data_path_with_root():
    """Test with root directory."""
    data_path = "seed_{seed}/{split}.npz"
    result = _resolve_data_path(data_path, split="train", seed=1337, root="root")
    assert result == Path("root/seed_1337/train.npz")


def test_resolve_data_path_missing_placeholder_error():
    """Test error when placeholder is missing."""
    data_path = "data/{split}.npz"
    with pytest.raises(ValueError, match=".*{seed}.*placeholder"):
        _resolve_data_path(data_path, seed=42, split="train")
