"""Tests for data loading utilities, including dataset and dataloader."""

from joint_improvement.utils.data.dataset import SequenceDataset


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
