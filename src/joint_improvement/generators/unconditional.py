"""Unconditional sequence generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from .generator import GeneratorMixin

if TYPE_CHECKING:
    from joint_improvement.hyformer.model import Hyformer


class UnconditionalGeneratorMixin(GeneratorMixin):
    """Unconditional sequence generation mixin."""

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample next token from logits.

        Parameters
        ----------
        logits : torch.Tensor
            Logits tensor of shape [B, vocab_size].
        temperature : float, default=1.0
            Temperature for unconditional generation. Lower values make the distribution more peaked.
        top_k : int | None, default=None
            If provided, only sample from top-k tokens.
        top_p : float | None, default=None
            If provided, use nucleus (top-p) unconditional generation. Should be in (0, 1].
        generator : torch.Generator | None, default=None
            Random number generator for reproducible unconditional generation. If None, uses default random state.

        Returns
        -------
        torch.Tensor
            Sampled token IDs of shape [B].
        """
        if temperature != 1.0:
            logits = self._scale_logits(logits, temperature)

        if top_k is not None and top_k > 0:
            logits = self._apply_top_k(logits, top_k)

        if top_p is not None and top_p < 1.0:
            logits = self._apply_top_p(logits, top_p)

        probs = self._compute_probs(logits)
        next_tokens = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

        return next_tokens

    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        eos_token_id: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        use_cache: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.LongTensor:
        """Generate sequences using unconditional generation.

        Parameters
        ----------
        input_ids : torch.LongTensor
            Input token IDs of shape [B, T] where B is batch size and T is sequence length.
        max_new_tokens : int
            Maximum number of new tokens to generate.
        eos_token_id : int
            End-of-sequence token ID. Generation stops when this token is generated.
        temperature : float, default=1.0
            Unconditional generation temperature. Lower values make the distribution more peaked.
        top_k : int | None, default=None
            If provided, only sample from top-k tokens.
        top_p : float | None, default=None
            If provided, use nucleus (top-p) unconditional generation. Should be in (0, 1].
        use_cache : bool, default=False
            Whether to use KV caching for faster generation.
        generator : torch.Generator | None, default=None
            Random number generator for reproducible generation. If None, uses default random state.

        Returns
        -------
        torch.LongTensor
            Generated sequences of shape [B, T + max_new_tokens].
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Initialize KV caches if using cache
        kv_caches = None
        if use_cache:
            kv_caches = [None] * self._get_num_layers()

        # Track generated sequences and whether each is finished
        generated = input_ids.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            # Get logits for next token (only use last token when using cache)
            if use_cache and kv_caches is not None:
                # When using cache, only pass the last token
                input_ids_for_logits = generated[:, -1:]
            else:
                input_ids_for_logits = generated

            logits, kv_caches = self._get_model_logits(input_ids_for_logits, kv_caches=kv_caches, use_cache=use_cache)

            # Sample next token
            next_tokens = self._sample_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            )

            # Update generated sequences
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)

            # Check for EOS tokens
            if eos_token_id is not None:
                finished = finished | (next_tokens == eos_token_id)
                if finished.all():
                    break

        return generated

    def generate_batch(
        self,
        input_ids: torch.LongTensor,
        num_samples: int,
        max_new_tokens: int,
        eos_token_id: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        use_cache: bool = True,
        seed: int | None = None,
    ) -> torch.LongTensor:
        """Generate multiple sequences using unconditional generation.

        Parameters
        ----------
        input_ids : torch.LongTensor
            Input token IDs of shape [B, T] where B is batch size and T is sequence length.
        num_samples : int
            Number of samples to generate for each input.
        max_new_tokens : int
            Maximum number of new tokens to generate.
        eos_token_id : int
            End-of-sequence token ID. Generation stops when this token is generated.
        temperature : float, default=1.0
            Unconditional generation temperature. Lower values make the distribution more peaked.
        top_k : int | None, default=None
            If provided, only sample from top-k tokens.
        top_p : float | None, default=None
            If provided, use nucleus (top-p) unconditional generation. Should be in (0, 1].
        use_cache : bool, default=True
            Whether to use KV caching for faster generation. Passed to generate().
        seed : int | None, default=None
            Random seed for reproducible generation. If provided, creates generators for each sample.

        Returns
        -------
        torch.LongTensor
            Generated sequences of shape [B * num_samples, T + max_new_tokens].
        """
        device = input_ids.device

        generated_sequences = []
        for sample_idx in tqdm(range(num_samples), desc="Generating sequences"):
            # Create generator from seed for each sample
            generator = None
            if seed is not None:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed + sample_idx)

            # Generate single sequence
            generated = self.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_cache=use_cache,
                generator=generator,
            )
            generated_sequences.append(generated)

        # Concatenate all generated sequences
        return torch.cat(generated_sequences, dim=0)

    @torch.inference_mode()
    def _get_model_logits(
        self: Hyformer,
        input_ids: torch.LongTensor,
        kv_caches: list | None = None,
        use_cache: bool = True,
    ) -> tuple[torch.FloatTensor, list | None]:
        """Get model logits for next token prediction.

        Works with Hyformer and other backbone models that follow the same interface.

        Parameters
        ----------
        input_ids : torch.LongTensor
            Input token IDs of shape [B, T].
        kv_caches : list | None, default=None
            List of KV caches for each layer.
        use_cache : bool, default=True
            Whether to use and update KV caches.

        Returns
        -------
        tuple[torch.FloatTensor, list | None]
            Tuple containing:
            - Logits tensor of shape [B, vocab_size] for the next token
            - Updated KV caches (or None if not using cache)
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=None,
            task="lm",
            kv_caches=kv_caches,
            use_cache=use_cache,
        )
        logits = outputs.logits  # [B, T, vocab_size] or [B, vocab_size]

        # Extract updated KV caches from extras
        updated_kv_caches = outputs.extras.get("kv_caches") if use_cache else None

        # Handle both [B, T, vocab_size] and [B, vocab_size] shapes
        if logits.dim() == 3:
            logits = logits[:, -1, :]  # [B, T, vocab_size] -> [B, vocab_size]
        # If already [B, vocab_size], use as is

        return logits, updated_kv_caches

    def _get_num_layers(self: Hyformer) -> int:
        """Get number of transformer layers for KV cache initialization.

        Returns
        -------
        int
            Number of layers in the model.
        """
        return len(self.blocks)
