"""Tests for LMCollator and Language Modeling task dataloader."""

import torch

from joint_improvement.hyformer.inputs import ModelInput
from joint_improvement.tokenizers import SMILESTokenizer
from joint_improvement.utils import SequenceDataLoader
from joint_improvement.utils.data.collators import LMCollator
from joint_improvement.utils.data.dataset import SequenceDataset


def test_lm_collator_basic():
    """Test basic functionality of LMCollator."""
    # Setup
    tokenizer_config_path = "configs/tokenizers/smiles/zinc250k/tokenizer_config.json"
    tokenizer = SMILESTokenizer.from_pretrained(tokenizer_config_path)

    sequences = ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O"]
    targets = [1.0, 1.0]

    dataset = SequenceDataset(sequences=sequences, targets=targets)

    # Create dataloader
    lm_loader = SequenceDataLoader(
        dataset=dataset,
        collator=LMCollator(tokenizer),
        batch_size=2,
        shuffle=True,
    )

    # Get batch
    batch = next(iter(lm_loader))

    # Assertions
    assert isinstance(batch, ModelInput)
    assert torch.equal(
        batch["input_ids"][0],
        torch.tensor([66, 62, 15, 15, 1, 13, 20, 2, 20, 58, 5, 58, 58, 58, 58, 58, 5, 15, 1, 13, 20, 2, 20, 63]),
    )
    assert torch.equal(
        batch["attention_mask"][0],
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    )
    assert batch["task"] == "lm"
    assert batch["targets"] is None
    assert torch.equal(batch["labels"][0], batch["input_ids"][0])
