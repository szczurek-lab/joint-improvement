"""Collator for masked language modeling (BERT-style) tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from joint_improvement.backbones.inputs import ModelInput
from joint_improvement.utils.data.dataset import SequenceDataset

from .base import BaseCollator

if TYPE_CHECKING:
    from joint_improvement.tokenizers import SMILESTokenizer


class MLMCollator(BaseCollator):
    """Collator for masked language modeling (BERT-style pretraining).

    Randomly masks tokens in the input sequence and prepares labels for predicting
    the masked tokens. Follows BERT-style masking strategy:
    - 15% of tokens are selected for masking
    - Of those 15%: 80% replaced with [MASK], 10% with random token, 10% unchanged
    - Labels: -100 for non-masked positions, original token ID for masked positions
    """

    def __init__(
        self,
        tokenizer: SMILESTokenizer,
        task_token: str = "<mlm>",
        max_length: int = 512,
        mlm_probability: float = 0.15,
    ) -> None:
        super().__init__(tokenizer, task_token=task_token, max_length=max_length)
        self.mlm_probability = mlm_probability

    def _mask_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply BERT-style masking to input tokens.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs of shape [B, T].
        attention_mask : torch.Tensor
            Attention mask of shape [B, T] (1 for real tokens, 0 for padding).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (masked_input_ids, labels) where:
            - masked_input_ids: Input with some tokens masked/replaced
            - labels: Labels tensor with -100 for non-masked positions and
              original token ID for masked positions.
        """
        labels = input_ids.clone()

        # Get special token IDs that should not be masked
        pad_token_id = self.tokenizer.pad_token_id
        mask_token_id = self.tokenizer.mask_token_id

        # Get task token ID (always required, validated during initialization)
        task_token_id = self.tokenizer.task_token_ids[self.task_token]

        # Collect all special token IDs that should not be masked
        special_token_ids = {pad_token_id}
        if mask_token_id is not None:
            special_token_ids.add(mask_token_id)
        special_token_ids.add(task_token_id)

        # Create probability matrix for masking (only for non-padding, non-special tokens)
        probability_matrix = torch.full(labels.shape, self.mlm_probability, device=input_ids.device)

        # Set probability to 0 for padding tokens and special tokens
        for special_id in special_token_ids:
            probability_matrix.masked_fill_(input_ids == special_id, 0.0)
        probability_matrix.masked_fill_(attention_mask == 0, 0.0)

        # Sample which tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # 80% of the time, replace masked input tokens with mask_token
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8, device=input_ids.device)).bool() & masked_indices
        )
        input_ids[indices_replaced] = mask_token_id

        # 10% of the time, replace masked input tokens with random word
        # Of the remaining masked tokens (not replaced with [MASK]),
        # randomly select 50% to replace with random tokens
        # This gives us: 10% / (10% + 10%) = 50% of the remaining 20%
        remaining_masked = masked_indices & ~indices_replaced
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5, device=input_ids.device)).bool() & remaining_masked
        )

        # Generate random words, excluding special tokens
        vocab_size = len(self.tokenizer.vocab)
        random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=input_ids.device)

        # Filter out special tokens from random replacements
        # If random word is a special token, keep the original token instead
        for special_id in special_token_ids:
            # For positions where random word equals special token, don't replace
            special_mask = (random_words == special_id) & indices_random
            # Keep original token for these positions (they'll remain unchanged)
            indices_random = indices_random & ~special_mask

        input_ids[indices_random] = random_words[indices_random]

        # The remaining 10% (masked_indices & ~indices_replaced & ~indices_random)
        # keep the original token unchanged

        return input_ids, labels

    def __call__(self, batch: list[dict[str, Any]]) -> ModelInput:
        """Collate a batch for masked language modeling.

        Adds task token, applies padding with truncation, creates attention masks,
        randomly masks tokens, and prepares labels for MLM.

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
        # This must happen BEFORE masking so task token is included in the sequence
        encoded, attention_mask, task_token_id = self._prepend_task_token(encoded, attention_mask)

        # Apply truncation
        encoded, attention_mask = self._apply_truncation(encoded, attention_mask)

        # Apply padding
        encoded, attention_mask = self._apply_padding(encoded, attention_mask)

        # Convert to tensors
        input_ids = torch.tensor(encoded, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)

        # Apply masking and prepare labels
        # Task token is already in input_ids and will be protected from masking
        masked_input_ids, labels = self._mask_tokens(input_ids, attention_mask_tensor)

        # Explicitly ensure task token position in labels is -100 (no loss computed)
        # Task token is at position 0, ensure label is -100
        labels[:, 0] = -100

        return self._to_model_input(
            input_ids=masked_input_ids,
            attention_mask=attention_mask_tensor,
            labels=labels,
            task=self.task_token,
        )
