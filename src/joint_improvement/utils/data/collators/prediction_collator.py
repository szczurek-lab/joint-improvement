"""Collator for prediction tasks (classification and regression)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from joint_improvement.backbones.inputs import ModelInput
from joint_improvement.utils.data.dataset import SequenceDataset

from .base import BaseCollator

if TYPE_CHECKING:
    from joint_improvement.tokenizers import SMILESTokenizer


class PredictionCollator(BaseCollator):
    """Collator for prediction tasks (classification or regression).

    Handles both discrete class labels (classification) and continuous values (regression).
    Automatically detects label type or uses specified dtype.
    """

    def __init__(
        self,
        tokenizer: SMILESTokenizer,
        task_token: str = "<prediction>",
        max_length: int = 128,
    ) -> None:
        super().__init__(tokenizer, task_token=task_token, max_length=max_length)

    def __call__(self, batch: list[dict[str, Any]]) -> ModelInput:
        """Collate a batch for prediction tasks.

        Adds task token, applies padding with truncation, creates attention masks,
        and prepares labels for classification or regression tasks.

        Parameters
        ----------
        batch : list[dict[str, Any]]
            List of samples, each containing "sequence" and "targets" keys.

        Returns
        -------
        ModelInput
            Model input container with input_ids, attention_mask, labels, and task.
        """
        sequences = [item[SequenceDataset.SEQUENCE_FIELD] for item in batch]
        targets = [item[SequenceDataset.TARGET_FIELD] for item in batch]

        tokenized = self.tokenizer(sequences)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Add task token first
        input_ids, attention_mask, task_token_id = self._prepend_task_token(input_ids, attention_mask)

        # Apply truncation
        input_ids, attention_mask = self._apply_truncation(input_ids, attention_mask)

        # Apply padding
        input_ids, attention_mask = self._apply_padding(input_ids, attention_mask)

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)

        # Determine label dtype and prepare labels
        label_dtype = torch.long if isinstance(targets[0], (int, bool)) else torch.float32
        labels_tensor = torch.tensor(targets, dtype=label_dtype)

        return self._to_model_input(
            input_ids=input_ids,
            attention_mask=attention_mask_tensor,
            labels=labels_tensor,
            task=self.task_token,
        )
