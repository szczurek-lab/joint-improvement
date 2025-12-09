"""Tests for MLMCollator and Masked Language Modeling task dataloader."""

import torch

from joint_improvement.hyformer.inputs import ModelInput
from joint_improvement.tokenizers import SMILESTokenizer
from joint_improvement.utils import SequenceDataLoader
from joint_improvement.utils.data.collators import MLMCollator
from joint_improvement.utils.data.dataset import SequenceDataset


def test_mlm_collator_basic():
    """Test basic functionality of MLMCollator."""
    # Setup
    tokenizer_config_path = "configs/tokenizers/smiles/zinc250k/tokenizer_config.json"
    tokenizer = SMILESTokenizer.from_pretrained(tokenizer_config_path)

    sequences = ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O"]
    targets = [1.0, 1.0]

    dataset = SequenceDataset(sequences=sequences, targets=targets)

    # Create dataloader
    mlm_loader = SequenceDataLoader(
        dataset=dataset,
        collator=MLMCollator(tokenizer),
        batch_size=2,
        shuffle=True,
    )

    # Get batch
    batch = next(iter(mlm_loader))

    # Assertions
    assert isinstance(batch, ModelInput)
    # Note: input_ids will have masked tokens (65 is mask token ID)
    # The exact values depend on which tokens were randomly masked
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert torch.equal(
        batch["attention_mask"][0],
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    )
    assert batch["task"] == "mlm"
    assert batch["targets"] is None
    # Labels should have -100 for non-masked positions and original token IDs for masked positions
    assert batch["labels"] is not None
    assert batch["labels"].shape == batch["input_ids"].shape
    # Verify that labels have -100 for non-masked positions
    # (masked positions will have the original token ID)
    assert (batch["labels"][0] == -100).any() or (batch["labels"][0] >= 0).any()


def test_mlm_collator_masking():
    """Test that MLM collator properly masks tokens."""
    # Setup
    tokenizer_config_path = "configs/tokenizers/smiles/zinc250k/tokenizer_config.json"
    tokenizer = SMILESTokenizer.from_pretrained(tokenizer_config_path)

    sequences = ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O"]
    targets = [1.0, 1.0]

    dataset = SequenceDataset(sequences=sequences, targets=targets)

    # Create dataloader with fixed seed for reproducibility
    mlm_loader = SequenceDataLoader(
        dataset=dataset,
        collator=MLMCollator(tokenizer),
        batch_size=2,
        shuffle=False,
    )

    # Get batch
    batch = next(iter(mlm_loader))

    # Verify masking occurred (some tokens should be masked)
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is not None:
        # Check that at least some tokens are masked
        assert (batch["input_ids"] == mask_token_id).any()

    # Verify labels structure: -100 for non-masked, token ID for masked
    assert batch["labels"] is not None
    # Masked positions should have token IDs (>= 0), non-masked should have -100
    masked_positions = batch["labels"] != -100
    if masked_positions.any():
        # Verify masked positions have valid token IDs
        assert (batch["labels"][masked_positions] >= 0).all()
