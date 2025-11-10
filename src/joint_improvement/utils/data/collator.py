"""Collator for multi-task sequence datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class MultiTaskSequenceCollator:
    """Collator for multi-task sequence datasets."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, task: str):
        self.tokenizer = tokenizer
        self.task = task

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        # TODO: Implement proper collation logic
        return batch[0] if batch else {}

        # It should store HF collators in a dictionary, and call the appropriate one based on the task.


# Type alias for backward compatibility
SequenceCollator = MultiTaskSequenceCollator
