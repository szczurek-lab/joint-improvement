"""Transformer block (encoder/decoder agnostic)."""

from __future__ import annotations

import torch
from torch import nn

from .attention import SelfAttention
from .kv_cache import KVCache
from .mlp import SwiGLUMLP
from .rms_norm import RMSNorm


class TransformerBlock(nn.Module):
    """
    Transformer block (encoder/decoder agnostic).

    Implements a transformer block with a self-attention sub-layer and an MLP
    sub-layer, each with pre-normalization using RMSNorm and residual connections.
    Can function as either an encoder or decoder block depending on the attention
    mask type:
    - Encoder mode: Uses bidirectional attention (all positions can attend to all)
    - Decoder mode: Uses causal attention (autoregressive, positions attend to previous)

    The behavior is controlled by the `is_causal` parameter and `attn_mask` in the
    forward pass.

    Parameters
    ----------
    d_model : int
        Model dimension (hidden size).
    n_heads : int
        Number of attention heads.
    max_seq_len : int
        Maximum sequence length for positional encodings and attention masks.
    attn_dropout : float, default=0.0
        Dropout probability for attention weights.
    resid_dropout : float, default=0.0
        Dropout probability for residual connections.
    eps : float, default=1e-6
        Epsilon value for RMSNorm layers.

    Attributes
    ----------
    input_norm : RMSNorm
        RMS normalization applied before attention.
    post_attn_norm : RMSNorm
        RMS normalization applied before MLP.
    attn : nn.Module
        Self-attention module.
    mlp : SwiGLUMLP
        SwiGLU MLP module.
    resid_dropout : nn.Dropout
        Dropout layer for residual connections.

    Notes
    -----
    The forward pass follows this structure:
        x -> x + Attn(RMSNorm(x))
        x -> x + MLP(RMSNorm(x))

    Each sub-layer uses pre-normalization (norm before the operation) and
    residual connections with optional dropout.

    Encoder vs Decoder behavior:
    - Set `is_causal=True` or provide a causal mask for decoder/autoregressive mode
    - Set `is_causal=False` or provide a bidirectional mask for encoder mode
    - The same block architecture works for both modes, only the attention pattern differs
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.input_norm = RMSNorm(d_model, eps=eps)
        self.post_attn_norm = RMSNorm(d_model, eps=eps)

        self.attn = SelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )
        self.mlp = SwiGLUMLP(d_model=d_model)

        self.resid_dropout = nn.Dropout(resid_dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
        is_causal: bool = True,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """
        Forward pass through the transformer block (encoder or decoder mode).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, T, D] where B is batch size,
            T is sequence length, and D is the model dimension.
        attn_mask : Optional[torch.Tensor], default=None
            Attention mask tensor. Can be boolean or additive mask,
            broadcastable to shape [B, H, T_q, T_k] where H is number
            of heads, T_q is query sequence length, and T_k is key
            sequence length.
        kv_cache : Optional[KVCache], default=None
            Optional key-value cache for autoregressive generation.
            If provided and use_cache=True, will be updated with new
            key-value pairs.
        use_cache : bool, default=False
            Whether to use and update the key-value cache.
        is_causal : bool, default=True
            Whether to apply causal masking in attention.
            - True: Decoder mode (autoregressive, causal attention)
            - False: Encoder mode (bidirectional attention)

        Returns
        -------
        tuple[torch.Tensor, Optional[KVCache]]
            Tuple containing:
            - Output tensor of shape [B, T, D]
            - Updated key-value cache (or None if not used)

        Notes
        -----
        The same transformer block can function as:
        - Encoder block: Set `is_causal=False` or provide bidirectional `attn_mask`
        - Decoder block: Set `is_causal=True` or provide causal `attn_mask`

        This allows the same architecture to be used for both generation (decoder)
        and prediction (encoder) tasks in Hyformer.
        """
        # ----- Attention sub-layer -----
        residual = x
        x_norm = self.input_norm(x)

        attn_out, kv_cache = self.attn(
            x_norm,
            attn_mask=attn_mask,
            kv_cache=kv_cache,
            use_cache=use_cache,
            is_causal=is_causal,
        )
        x = residual + self.resid_dropout(attn_out)

        # ----- MLP sub-layer -----
        residual = x
        x_norm = self.post_attn_norm(x)
        mlp_out = self.mlp(x_norm)
        x = residual + self.resid_dropout(mlp_out)

        return x, kv_cache
