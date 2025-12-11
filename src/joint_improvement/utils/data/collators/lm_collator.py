"""Collator for causal language modeling tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .base import BaseCollatorWithPadding

if TYPE_CHECKING:
    from joint_improvement.tokenizers import SMILESTokenizer


class LMCollator(BaseCollatorWithPadding):
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

    def _prepare_labels_and_targets(
        self,
        batch: list[dict[str, Any]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Prepare labels for causal language modeling.

        Labels are the same as input_ids (will be shifted internally during loss computation).
        Task token is included in labels.

        Parameters
        ----------
        batch : list[dict[str, Any]]
            Original batch (unused, kept for interface consistency).
        input_ids : torch.Tensor
            Input token IDs tensor.
        attention_mask : torch.Tensor
            Attention mask tensor (unused, kept for interface consistency).

        Returns
        -------
        tuple[torch.Tensor | None, torch.Tensor | None]
            Tuple of (labels, None) where labels are cloned input_ids.
        """
        # Prepare labels (same as input_ids for causal LM, will be shifted in loss computation)
        labels = input_ids.clone()
        return labels, None
