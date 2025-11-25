"""Collator for causal language modeling tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from joint_improvement.backbones.inputs import ModelInput
from joint_improvement.utils.data.dataset import SequenceDataset

from .base import BaseCollator

if TYPE_CHECKING:
    from joint_improvement.tokenizers import SMILESTokenizer


class LMCollator(BaseCollator):
    """Collator for causal language modeling (next-token prediction).

    Labels are set to be the same as input_ids (shifted internally during loss computation).
    """

    def __init__(
        self,
        tokenizer: SMILESTokenizer,
        task_token: str = "lm",
        max_length: int = 128,
    ) -> None:
        super().__init__(tokenizer, task_token=task_token, max_length=max_length)

    def __call__(self, batch: list[dict[str, Any]]) -> ModelInput:
        """Collate a batch for language modeling.

        Adds task token, applies padding with truncation, creates attention masks,
        and prepares labels for causal language modeling.

        Parameters
        ----------
        batch : list[dict[str, Any]]
            List of samples, each containing a "sequence" key.

        Returns
        -------
        ModelInput
            Model input container with input_ids, attention_mask, labels, and task.
        """
        sequences = [item[SequenceDataset.SEQUENCE_FIELD] for item in batch]

        tokenized = self.tokenizer(sequences)
        encoded = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Add task token first (before truncation and padding)
        encoded, attention_mask, task_token_id = self._prepend_task_token(encoded, attention_mask)

        # Apply truncation
        encoded, attention_mask = self._apply_truncation(encoded, attention_mask)

        # Apply padding
        encoded, attention_mask = self._apply_padding(encoded, attention_mask)

        # Convert to tensors
        input_ids = torch.tensor(encoded, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)

        # Prepare labels (same as input_ids for causal LM, will be shifted in loss computation)
        # Task token is included in labels (will be shifted during loss computation)
        labels = input_ids.clone()

        return self._to_model_input(
            input_ids=input_ids,
            attention_mask=attention_mask_tensor,
            labels=labels,
            task=self.task_token,
        )
