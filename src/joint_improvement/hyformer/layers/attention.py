"""Self-attention module with rotary positional embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .kv_cache import KVCache
from .rotary import RotaryPositionalEmbedding


class SelfAttention(nn.Module):
    """
    Self-attention module with rotary positional embeddings.

    Parameters
    ----------
    d_model : int
        Model dimension (hidden size). Must be divisible by n_heads.
    n_heads : int
        Number of attention heads.
    max_seq_len : int
        Maximum sequence length for positional encodings and KV cache.
    attn_dropout : float, default=0.0
        Dropout probability for attention weights.

    Attributes
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    head_dim : int
        Dimension of each attention head (d_model // n_heads).
    max_seq_len : int
        Maximum sequence length.
    q_proj : nn.Linear
        Query projection layer, shape [d_model, d_model].
    k_proj : nn.Linear
        Key projection layer, shape [d_model, d_model].
    v_proj : nn.Linear
        Value projection layer, shape [d_model, d_model].
    o_proj : nn.Linear
        Output projection layer, shape [d_model, d_model].
    attn_dropout_p : float
        Attention dropout probability.
    rotary : RotaryPositionalEmbedding
        Rotary positional embedding module.

    Notes
    -----
    The attention mechanism supports both causal (decoder) and bidirectional
    (encoder) modes via the `is_causal` parameter and `attn_mask`.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = attn_dropout
        self.rotary = RotaryPositionalEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool,
        attn_mask: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        use_cache: bool = False
    ) -> tuple[torch.Tensor, KVCache | None]:
        """
        Forward pass through self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, T_q, D] where B is batch size,
            T_q is query sequence length, and D is model dimension.
        attn_mask : Optional[torch.Tensor], default=None
            Attention mask tensor. Can be boolean or additive mask,
            broadcastable to shape [B, H, T_q, T_k] where H is number
            of heads, T_q is query sequence length, and T_k is key
            sequence length.
        kv_cache : Optional[KVCache], default=None
            Optional key-value cache for autoregressive generation.
            If None and use_cache=True, a new cache will be created.
        use_cache : bool, default=False
            Whether to use and update key-value cache.
        is_causal : bool, default=True
            Whether to apply causal masking in attention.
            - True: Causal/autoregressive attention (decoder mode)
            - False: Bidirectional attention (encoder mode)

        Returns
        -------
        tuple[torch.Tensor, Optional[KVCache]]
            Tuple containing:
            - Output tensor of shape [B, T_q, D]
            - Updated key-value cache (or None if not used)
        """
        
        if is_causal and attn_mask is not None:
            raise ValueError("Causal attention does not support an explicit attention mask.")
        if not is_causal and attn_mask is None:
            raise ValueError("Bidirectional attention requires an attention mask.")
        
        if attn_mask is not None:
            # Convert [B, T] padding mask (1=keep, 0=pad) -> SDPA bool mask [B, 1, 1, T] (True=mask)
            attn_mask = (attn_mask == 0).unsqueeze(1).unsqueeze(2)
        
        # Get batch size and query sequence length
        B, T_q, _ = x.shape

        # Project to Q, K, V
        q_lin = self.q_proj(x)
        k_lin = self.k_proj(x)
        v_lin = self.v_proj(x)

        # Reshape to [B, T, H, Hd] for rotary positional embeddings
        q_seq_head = q_lin.view(B, T_q, self.n_heads, self.head_dim)
        k_seq_head = k_lin.view(B, T_q, self.n_heads, self.head_dim)

        # Compute offset for rotary embeddings (for incremental generation)
        seq_offset = kv_cache.length if (use_cache and kv_cache is not None) else 0

        # Apply rotary positional embeddings
        q_rot = self.rotary(q_seq_head, offset=seq_offset)
        k_rot = self.rotary(k_seq_head, offset=seq_offset)

        # Convert to head-major format [B, H, T, Hd] for SDPA and cache
        q = q_rot.transpose(1, 2)  # [B, H, T_q, Hd]
        k_new = k_rot.transpose(1, 2)  # [B, H, T_q, Hd]
        v_new = v_lin.view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T_q, Hd]

        # KV cache update
        if use_cache:
            if kv_cache is None:
                kv_cache = KVCache(
                    batch_size=B,
                    num_heads=self.n_heads,
                    head_dim=self.head_dim,
                    max_seq_len=self.max_seq_len,
                    dtype=x.dtype,
                    device=x.device,
                )
            kv_cache.append(k_new, v_new)
            k, v = kv_cache.get_kv()  # Full history
        else:
            k, v = k_new, v_new

        # When using a KV cache, query positions are offset by seq_offset.
        # PyTorch SDPA's built-in `is_causal=True` assumes queries start at 0,
        # so for seq_offset > 0 we must avoid the built-in causal path. For
        # multi-token chunks (T_q > 1), we additionally apply an offset-aware
        # causal mask to prevent looking ahead within the chunk.
        if is_causal and use_cache and seq_offset > 0 and T_q > 1:
            T_k = k.shape[-2]
            q_pos = (seq_offset + torch.arange(T_q, device=x.device)).unsqueeze(1)  # [T_q, 1]
            k_pos = torch.arange(T_k, device=x.device).unsqueeze(0)  # [1, T_k]
            attn_mask = k_pos > q_pos  # bool mask; True means "mask out"

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=is_causal,
        )  # [B, H, T_q, Hd]

        # Merge heads and project output
        y = y.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        y = self.o_proj(y)

        return y, kv_cache
