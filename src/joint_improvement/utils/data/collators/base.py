"""Base collator interface for task-specific collators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

from joint_improvement.backbones.inputs import ModelInput

if TYPE_CHECKING:
    from joint_improvement.tokenizers import SMILESTokenizer


class BaseCollator(ABC):
    """Base interface for task-specific collators."""

    def __init__(
        self,
        tokenizer: SMILESTokenizer,
        task_token: str,
        max_length: int = 128,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_token = task_token

    def _apply_truncation(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Apply truncation to encoded sequences and attention masks.

        Parameters
        ----------
        input_ids : list[list[int]]
            List of encoded sequences.
        attention_mask : list[list[int]]
            List of attention masks.

        Returns
        -------
        tuple[list[list[int]], list[list[int]]]
            Truncated encoded sequences and attention masks.
        """
        input_ids = [seq[: self.max_length] for seq in input_ids]
        attention_mask = [mask[: self.max_length] for mask in attention_mask]
        return input_ids, attention_mask

    def _apply_padding(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Apply padding to encoded sequences and attention masks.

        Parameters
        ----------
        encoded : list[list[int]]
            List of encoded sequences.
        attention_mask : list[list[int]]
            List of attention masks.

        Returns
        -------
        tuple[list[list[int]], list[list[int]]]
            Padded encoded sequences and attention masks.
        """
        pad_token_id = self.tokenizer.pad_token_id
        max_len = max(len(seq) for seq in input_ids)

        input_ids = [seq + [pad_token_id] * (max_len - len(seq)) for seq in input_ids]
        attention_mask = [mask + [0] * (max_len - len(mask)) for mask in attention_mask]
        return input_ids, attention_mask

    def _prepend_task_token(
        self, encoded: list[list[int]], attention_mask: list[list[int]]
    ) -> tuple[list[list[int]], list[list[int]], int | None]:
        """Prepend task token to encoded sequences and update attention_mask.

        Adds the task token at the beginning of each sequence and ensures
        the attention mask includes the task token (with value 1, meaning
        it's not masked out).

        Parameters
        ----------
        encoded : list[list[int]]
            List of encoded sequences (each is a list of token IDs).
        attention_mask : list[list[int]]
            List of attention masks (each is a list of 1s and 0s).

        Returns
        -------
        tuple[list[list[int]], list[list[int]], int | None]
            Tuple of (encoded_with_task_token, attention_mask_with_task_token, task_token_id).
            If no task token, task_token_id is None and sequences are unchanged.
        """
        if not self.task_token or not hasattr(self.tokenizer, "task_token_ids"):
            return encoded, attention_mask, None

        task_token_id = self.tokenizer.task_token_ids.get(self.task_token)
        if task_token_id is None:
            return encoded, attention_mask, None

        return (
            [[task_token_id] + seq for seq in encoded],
            [[1] + mask for mask in attention_mask],
            task_token_id,
        )

    def _to_model_input(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: str,
        labels: torch.Tensor | None = None,
    ) -> ModelInput:
        """Convert tokenized tensors to ModelInput.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs.
        attention_mask : torch.Tensor
            Attention mask.
        task : str
            Task name.
        labels : torch.Tensor | None
            Labels tensor.

        Returns
        -------
        ModelInput
            Model input container.
        """
        return ModelInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            task=task,
        )

    @abstractmethod
    def __call__(self, batch: list[dict[str, Any]]) -> ModelInput:
        """Collate a batch of samples from SequenceDataset.

        Returns
        -------
        ModelInput
            Model input container with input_ids, attention_mask, labels, and task.
        """
        pass
