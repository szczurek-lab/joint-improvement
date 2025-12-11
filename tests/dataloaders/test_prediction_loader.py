"""Tests for PredictionCollator and Prediction task dataloader."""

import torch

from joint_improvement.hyformer.inputs import ModelInput
from joint_improvement.tokenizers import SMILESTokenizer
from joint_improvement.utils import SequenceDataLoader
from joint_improvement.utils.data.collators import PredictionCollator
from joint_improvement.utils.data.dataset import SequenceDataset


def test_prediction_collator_basic():
    """Test basic functionality of PredictionCollator."""
    # Setup
    tokenizer_config_path = "configs/tokenizers/smiles/zinc250k/tokenizer_config.json"
    tokenizer = SMILESTokenizer.from_pretrained(tokenizer_config_path)

    sequences = ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O"]
    targets = [1.0, 1.0]

    dataset = SequenceDataset(sequences=sequences, targets=targets)

    # Create dataloader
    prediction_loader = SequenceDataLoader(
        dataset=dataset,
        collator=PredictionCollator(tokenizer),
        batch_size=2,
        shuffle=True,
    )

    # Get batch
    batch = next(iter(prediction_loader))

    # Assertions
    assert isinstance(batch, ModelInput)
    assert torch.equal(
        batch["input_ids"][0],
        torch.tensor([69, 62, 15, 15, 1, 13, 20, 2, 20, 58, 5, 58, 58, 58, 58, 58, 5, 15, 1, 13, 20, 2, 20, 63]),
    )
    assert torch.equal(
        batch["attention_mask"][0],
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    )
    assert batch["task"] == "prediction"
    assert torch.equal(batch["targets"][0], torch.tensor([1.0]))
    assert batch["labels"] is None
