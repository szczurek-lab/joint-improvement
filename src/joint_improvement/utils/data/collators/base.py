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
        """Initialize base collator.

        Parameters
        ----------
        tokenizer : SMILESTokenizer
            Tokenizer instance that contains task_tokens and task_token_ids mappings.
        task_token : str
            Task token name (key in tokenizer.task_tokens), required and cannot be empty.
            For example, use "lm" not "<lm>" or the token ID. The tokenizer will convert
            the name to the actual token ID internally.
        max_length : int, default=128
            Maximum sequence length for truncation.

        Raises
        ------
        ValueError
            If task_token is empty, tokenizer doesn't have task_tokens, or task_token
            is not found in tokenizer.task_tokens.
        """
        if not task_token:
            raise ValueError("task_token is required and cannot be empty")

        if not hasattr(tokenizer, "task_tokens") or not tokenizer.task_tokens:
            raise ValueError("Tokenizer must have task_tokens defined")

        if task_token not in tokenizer.task_tokens:
            available_tokens = list(tokenizer.task_tokens.keys())
            raise ValueError(
                f"Task token '{task_token}' not found in tokenizer.task_tokens. "
                f"Available task tokens: {available_tokens}"
            )

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
    ) -> tuple[list[list[int]], list[list[int]], int]:
        """Prepend task token to encoded sequences and update attention_mask.

        Adds the task token at the beginning of each sequence and ensures
        the attention mask includes the task token (with value 1, meaning
        it's not masked out).

        The task token name is converted to a token ID by looking it up in
        tokenizer.task_token_ids. Task token is always required and validated
        during initialization.

        Parameters
        ----------
        encoded : list[list[int]]
            List of encoded sequences (each is a list of token IDs).
        attention_mask : list[list[int]]
            List of attention masks (each is a list of 1s and 0s).

        Returns
        -------
        tuple[list[list[int]], list[list[int]], int]
            Tuple of (encoded_with_task_token, attention_mask_with_task_token, task_token_id).
        """
        # Task token is always required and validated during __init__, so we can directly access it
        task_token_id = self.tokenizer.task_token_ids[self.task_token]

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
        targets: torch.Tensor | None = None,
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
            Labels tensor (for sequence-level tasks like MLM/LM).
        targets : torch.Tensor | None
            Targets tensor (for prediction tasks like regression/classification).

        Returns
        -------
        ModelInput
            Model input container.
        """
        return ModelInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            targets=targets,
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
