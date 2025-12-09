"""Tests for invalid SMILES handling in dataloaders."""

import pytest

from joint_improvement.tokenizers import SMILESTokenizer
from joint_improvement.utils import SequenceDataLoader
from joint_improvement.utils.data.collators import PredictionCollator
from joint_improvement.utils.data.dataset import SequenceDataset


def test_invalid_smiles_raises_error():
    """Test that invalid SMILES with unknown tokens raise an error during tokenization."""
    # Setup
    tokenizer_config_path = "configs/tokenizers/smiles/zinc250k/tokenizer_config.json"
    tokenizer = SMILESTokenizer.from_pretrained(tokenizer_config_path)

    # Invalid SMILES with '*' token (not in vocabulary)
    invalid_sequences = ["CC(=O)Oc1ccccc1C(=O)O*", "CC(=O)O"]
    targets = [1.0, 1.0]

    dataset = SequenceDataset(sequences=invalid_sequences, targets=targets)

    # Create dataloader
    prediction_loader = SequenceDataLoader(
        dataset=dataset,
        collator=PredictionCollator(tokenizer),
        batch_size=2,
        shuffle=False,
    )

    # Attempting to get a batch should raise KeyError due to unknown token '*'
    with pytest.raises(KeyError, match="Token '\\*' not found in vocabulary"):
        _ = next(iter(prediction_loader))


def test_valid_smiles_works():
    """Test that valid SMILES work correctly."""
    # Setup
    tokenizer_config_path = "configs/tokenizers/smiles/zinc250k/tokenizer_config.json"
    tokenizer = SMILESTokenizer.from_pretrained(tokenizer_config_path)

    # Valid SMILES
    valid_sequences = ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)O"]
    targets = [1.0, 1.0]

    dataset = SequenceDataset(sequences=valid_sequences, targets=targets)

    # Create dataloader
    prediction_loader = SequenceDataLoader(
        dataset=dataset,
        collator=PredictionCollator(tokenizer),
        batch_size=2,
        shuffle=False,
    )

    # Should work without errors
    batch = next(iter(prediction_loader))
    assert batch is not None
    assert batch["input_ids"].shape[0] == 2
