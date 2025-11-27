"""Collator for masked language modeling (BERT-style) tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .base import BaseCollatorWithPadding

if TYPE_CHECKING:
    from joint_improvement.tokenizers import SMILESTokenizer


class MLMCollator(BaseCollatorWithPadding):
    """Collator for masked language modeling (BERT-style pretraining).

    Randomly masks tokens in the input sequence and prepares labels for predicting
    the masked tokens. Follows BERT-style masking strategy with UL2-inspired variable
    masking rates:
    - Masking probability is sampled per sequence from a truncated normal distribution
    - Of masked tokens: 80% replaced with [MASK], 10% with random token, 10% unchanged
    - Labels: -100 for non-masked positions, original token ID for masked positions

    The truncated normal sampling introduces controlled variability in masking rates,
    similar to UL2's Mixture of Denoisers approach, which helps the model learn from
    diverse masking patterns during pretraining.
    """

    def __init__(
        self,
        tokenizer: SMILESTokenizer,
        task_token: str = "mlm",
        max_length: int = 128,
        mlm_probability: float | None = None,
        mlm_probability_mean: float = 0.15,
        mlm_probability_std: float = 0.08,
        mlm_probability_min: float = 0.05,
        mlm_probability_max: float = 0.5,
        use_truncated_normal: bool = True,
    ) -> None:
        """Initialize MLM collator.

        Parameters
        ----------
        tokenizer : SMILESTokenizer
            Tokenizer instance.
        task_token : str, default="mlm"
            Task token name.
        max_length : int, default=128
            Maximum sequence length.
        mlm_probability : float | None, default=None
            Fixed masking probability (for backward compatibility).
            If provided, overrides truncated normal sampling.
        mlm_probability_mean : float, default=0.15
            Mean of truncated normal distribution for masking probability.
            Inspired by UL2's R-Denoiser (15% masking rate). This is the standard
            BERT masking rate, proven effective for language model pretraining.
        mlm_probability_std : float, default=0.08
            Standard deviation of truncated normal distribution. With mean=0.15,
            this provides good variability: ~68% of sequences mask 7-23% of tokens,
            ~95% mask 5-31% of tokens, enabling diverse denoising patterns.
        mlm_probability_min : float, default=0.05
            Minimum masking probability (lower bound of truncation). Ensures some
            masking always occurs, preventing zero-mask batches that hurt learning.
        mlm_probability_max : float, default=0.5
            Maximum masking probability (upper bound of truncation).
            Inspired by UL2's X-Denoiser (50% masking rate), enabling extreme
            denoising scenarios that improve model robustness.
        use_truncated_normal : bool, default=True
            If True, sample masking probability from truncated normal per sequence.
            If False, use fixed mlm_probability (or mlm_probability_mean if mlm_probability is None).
        """
        super().__init__(tokenizer, task_token=task_token, max_length=max_length)

        if mlm_probability is not None:
            # Backward compatibility: use fixed probability
            self.mlm_probability = mlm_probability
            self.use_truncated_normal = False
        else:
            self.mlm_probability_mean = mlm_probability_mean
            self.mlm_probability_std = mlm_probability_std
            self.mlm_probability_min = mlm_probability_min
            self.mlm_probability_max = mlm_probability_max
            self.use_truncated_normal = use_truncated_normal
            # For backward compatibility, also set mlm_probability to mean
            self.mlm_probability = mlm_probability_mean

    def _sample_masking_probability(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample masking probabilities from truncated normal distribution.

        Implements UL2-inspired variable masking by sampling per-sequence probabilities
        from a truncated normal distribution. This introduces controlled variability
        while maintaining a target mean masking rate.

        Parameters
        ----------
        batch_size : int
            Number of sequences in batch.
        device : torch.device
            Device to place sampled probabilities on.

        Returns
        -------
        torch.Tensor
            Tensor of shape [batch_size] with masking probabilities per sequence.
        """
        if not self.use_truncated_normal:
            # Return fixed probability for all sequences
            return torch.full((batch_size,), self.mlm_probability, device=device)

        # Sample from normal distribution and clamp to bounds (truncated normal approximation)
        # This is an efficient approximation that works well in practice
        samples = torch.normal(
            mean=self.mlm_probability_mean,
            std=self.mlm_probability_std,
            size=(batch_size,),
            device=device,
        )
        # Clamp to bounds (truncation)
        samples = torch.clamp(samples, min=self.mlm_probability_min, max=self.mlm_probability_max)

        return samples

    def _mask_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply BERT-style masking to input tokens and prepare labels.

        Modifies input_ids in place and returns labels tensor. Uses UL2-inspired variable
        masking rates sampled from a truncated normal distribution per sequence:
        - Masking probability sampled per sequence from truncated normal
        - Of masked tokens: 80% replaced with [MASK], 10% with random token, 10% unchanged
        - Labels: -100 for non-masked positions, original token ID for masked positions

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs of shape [B, T] (modified in place).
        attention_mask : torch.Tensor
            Attention mask of shape [B, T] (1 for real tokens, 0 for padding).

        Returns
        -------
        torch.Tensor
            Labels tensor with -100 for non-masked positions and original token ID
            for masked positions. Task token position is set to -100.
        """
        labels = input_ids.clone()
        device = input_ids.device
        shape = input_ids.shape
        batch_size = shape[0]

        # Collect special token IDs that should not be masked
        special_token_ids = {
            self.tokenizer.pad_token_id,
            self.tokenizer.task_token_ids[self.task_token],
        }
        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id is not None:
            special_token_ids.add(mask_token_id)

        # Create mask for tokens that can be masked (not padding, not special, real tokens)
        can_mask = attention_mask.bool()
        for special_id in special_token_ids:
            can_mask = can_mask & (input_ids != special_id)

        # Sample masking probabilities per sequence from truncated normal (UL2-inspired)
        masking_probs = self._sample_masking_probability(batch_size, device)

        # Apply per-sequence masking probabilities
        # Create probability matrix: [B, T] where each row uses the sequence's sampled probability
        probability_matrix = masking_probs.unsqueeze(1).expand(-1, shape[1])

        # Sample which tokens to mask using per-sequence probabilities
        masked_indices = can_mask & (torch.rand(shape, device=device) < probability_matrix)
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # 80% of masked tokens: replace with [MASK]
        replace_with_mask = masked_indices & (torch.rand(shape, device=device) < 0.8)
        input_ids[replace_with_mask] = mask_token_id

        # 10% of masked tokens: replace with random token
        # (50% of remaining masked tokens after [MASK] replacement)
        remaining_masked = masked_indices & ~replace_with_mask
        replace_with_random = remaining_masked & (torch.rand(shape, device=device) < 0.5)

        if replace_with_random.any():
            # Generate random tokens, excluding special tokens
            vocab_size = len(self.tokenizer.vocab)
            random_tokens = torch.randint(vocab_size, shape, dtype=torch.long, device=device)

            # Filter out special tokens from random replacements
            for special_id in special_token_ids:
                replace_with_random = replace_with_random & (random_tokens != special_id)

            input_ids[replace_with_random] = random_tokens[replace_with_random]

        # Remaining 10% of masked tokens keep original token (already handled)
        # Task token is already protected from masking and has -100 label via ~masked_indices

        return labels

    def _prepare_labels_and_targets(
        self,
        batch: list[dict[str, Any]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Prepare labels for masked language modeling.

        Applies BERT-style masking to input_ids (in place) and prepares labels.

        Parameters
        ----------
        batch : list[dict[str, Any]]
            Original batch (unused, kept for interface consistency).
        input_ids : torch.Tensor
            Input token IDs tensor (modified in place by masking).
        attention_mask : torch.Tensor
            Attention mask tensor.

        Returns
        -------
        tuple[torch.Tensor | None, torch.Tensor | None]
            Tuple of (labels, None) where labels have -100 for non-masked positions.
        """
        labels = self._mask_tokens(input_ids, attention_mask)
        return labels, None
