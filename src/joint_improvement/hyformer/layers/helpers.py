"""Helper functions for creating attention masks."""

from __future__ import annotations

import torch


def create_causal_mask(
    seq_len: int,
    batch_size: int | None = None,
    num_heads: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create a causal (lower triangular) attention mask.

    Creates a mask where each position can only attend to previous positions
    and itself, enforcing autoregressive behavior. The mask has values of 0
    for positions that can attend and -inf (or large negative value) for
    positions that cannot attend.

    Parameters
    ----------
    seq_len : int
        Sequence length (both query and key sequence length).
    batch_size : Optional[int], default=None
        Batch size. If provided, adds batch dimension to the mask.
        If None, mask shape will be [seq_len, seq_len] or [1, seq_len, seq_len].
    num_heads : Optional[int], default=None
        Number of attention heads. If provided, adds head dimension.
        If None, mask shape will be [seq_len, seq_len] or [B, seq_len, seq_len].
    device : Optional[torch.device], default=None
        Device to create the mask on. If None, uses CPU.
    dtype : torch.dtype, default=torch.float32
        Data type for the mask tensor.

    Returns
    -------
    torch.Tensor
        Causal attention mask. Shape depends on provided dimensions:
        - If batch_size and num_heads are None: [seq_len, seq_len]
        - If batch_size provided: [batch_size, seq_len, seq_len]
        - If batch_size and num_heads provided: [batch_size, num_heads, seq_len, seq_len]
        - If only num_heads provided: [1, num_heads, seq_len, seq_len]

        Values are 0.0 for allowed positions and -inf for masked positions.

    Examples
    --------
    >>> # Basic causal mask
    >>> mask = create_causal_mask(seq_len=5)
    >>> mask.shape
    torch.Size([5, 5])

    >>> # With batch and heads
    >>> mask = create_causal_mask(seq_len=10, batch_size=2, num_heads=8)
    >>> mask.shape
    torch.Size([2, 8, 10, 10])

    Notes
    -----
    The mask is lower triangular (including diagonal), meaning position i
    can attend to positions j where j <= i. This enforces causal/autoregressive
    behavior where tokens can only see previous tokens.
    """
    if device is None:
        device = torch.device("cpu")

    # Create lower triangular mask (causal)
    # Shape: [seq_len, seq_len]
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=dtype), diagonal=1)
    # Invert: 0 where allowed, 1 where masked
    mask = mask.masked_fill(mask == 1, float("-inf"))
    # Now mask is 0 for allowed positions, -inf for masked positions

    # Add dimensions as needed
    if num_heads is not None:
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        if batch_size is not None:
            mask = mask.expand(batch_size, num_heads, seq_len, seq_len)
        else:
            mask = mask.expand(1, num_heads, seq_len, seq_len)
    elif batch_size is not None:
        mask = mask.unsqueeze(0)  # [1, seq_len, seq_len]
        mask = mask.expand(batch_size, seq_len, seq_len)

    return mask


def create_bidirectional_mask(
    seq_len: int,
    padding_mask: torch.Tensor | None = None,
    batch_size: int | None = None,
    num_heads: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create a bidirectional attention mask.

    Creates a mask where all positions can attend to all other positions
    (bidirectional attention). Optionally incorporates a padding mask to
    prevent attention to padding tokens.

    Parameters
    ----------
    seq_len : int
        Sequence length (both query and key sequence length).
    padding_mask : Optional[torch.Tensor], default=None
        Padding mask of shape [B, T] where True/1 indicates valid tokens
        and False/0 indicates padding tokens. If provided, batch_size
        will be inferred from this tensor.
    batch_size : Optional[int], default=None
        Batch size. Required if padding_mask is None. If padding_mask is
        provided, this parameter is ignored and batch_size is inferred
        from padding_mask.
    num_heads : Optional[int], default=None
        Number of attention heads. If provided, adds head dimension.
    device : Optional[torch.device], default=None
        Device to create the mask on. If None, uses device from padding_mask
        or CPU if padding_mask is None.
    dtype : torch.dtype, default=torch.float32
        Data type for the mask tensor.

    Returns
    -------
    torch.Tensor
        Bidirectional attention mask. Shape depends on provided dimensions:
        - If padding_mask provided: [B, seq_len, seq_len] or [B, num_heads, seq_len, seq_len]
        - If batch_size provided: [batch_size, seq_len, seq_len] or [batch_size, num_heads, seq_len, seq_len]
        - If neither: [seq_len, seq_len] or [num_heads, seq_len, seq_len] (if num_heads provided)

        Values are 0.0 for allowed positions and -inf for masked positions.

    Examples
    --------
    >>> # Basic bidirectional mask (all positions can attend)
    >>> mask = create_bidirectional_mask(seq_len=5)
    >>> mask.shape
    torch.Size([5, 5])
    >>> # All zeros (no masking)

    >>> # With padding mask
    >>> padding = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    >>> mask = create_bidirectional_mask(seq_len=5, padding_mask=padding)
    >>> mask.shape
    torch.Size([2, 5, 5])
    >>> # Padding positions (0s) will be masked with -inf

    >>> # With batch and heads
    >>> mask = create_bidirectional_mask(seq_len=10, batch_size=2, num_heads=8)
    >>> mask.shape
    torch.Size([2, 8, 10, 10])

    Notes
    -----
    If padding_mask is provided, positions where padding_mask is False/0
    will be masked (set to -inf) in both query and key dimensions.
    This prevents attention to and from padding tokens.
    """
    # Determine batch_size and device
    if padding_mask is not None:
        batch_size = padding_mask.shape[0]
        if device is None:
            device = padding_mask.device
        if padding_mask.dtype != torch.bool:
            padding_mask = padding_mask.bool()
        remove_batch_dim = False
    else:
        if batch_size is None:
            # No batch dimension if neither padding_mask nor batch_size provided
            batch_size = 1
            remove_batch_dim = True
        else:
            remove_batch_dim = False
        if device is None:
            device = torch.device("cpu")

    # Start with all zeros (no masking) for bidirectional attention
    # Shape: [batch_size, seq_len, seq_len]
    mask = torch.zeros(batch_size, seq_len, seq_len, device=device, dtype=dtype)

    # Apply padding mask if provided
    if padding_mask is not None:
        # padding_mask: [B, T]
        # Expand to [B, 1, T] and [B, T, 1] for broadcasting
        # Mask positions where either query or key is padding
        padding_mask_expanded_q = padding_mask.unsqueeze(1)  # [B, 1, T]
        padding_mask_expanded_k = padding_mask.unsqueeze(2)  # [B, T, 1]
        # Combined mask: True where both query and key are valid
        valid_mask = padding_mask_expanded_q & padding_mask_expanded_k  # [B, T, T]
        # Set -inf where invalid (either query or key is padding)
        mask = mask.masked_fill(~valid_mask, float("-inf"))

    # Add head dimension if needed
    if num_heads is not None:
        mask = mask.unsqueeze(1)  # [B, 1, seq_len, seq_len]
        mask = mask.expand(batch_size, num_heads, seq_len, seq_len)

    # Remove batch dimension if it was added as a placeholder
    if remove_batch_dim:
        if num_heads is not None:
            mask = mask.squeeze(0)  # [num_heads, seq_len, seq_len]
        else:
            mask = mask.squeeze(0)  # [seq_len, seq_len]

    return mask


def create_attention_mask(
    seq_len: int,
    is_causal: bool = True,
    padding_mask: torch.Tensor | None = None,
    batch_size: int | None = None,
    num_heads: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create an attention mask (causal or bidirectional).

    Convenience function that creates either a causal or bidirectional mask
    based on the is_causal parameter.

    Parameters
    ----------
    seq_len : int
        Sequence length (both query and key sequence length).
    is_causal : bool, default=True
        If True, creates a causal mask. If False, creates a bidirectional mask.
    padding_mask : Optional[torch.Tensor], default=None
        Padding mask of shape [B, T] for bidirectional attention.
        Only used when is_causal=False.
    batch_size : Optional[int], default=None
        Batch size. Required if padding_mask is None.
    num_heads : Optional[int], default=None
        Number of attention heads. If provided, adds head dimension.
    device : Optional[torch.device], default=None
        Device to create the mask on.
    dtype : torch.dtype, default=torch.float32
        Data type for the mask tensor.

    Returns
    -------
    torch.Tensor
        Attention mask with appropriate shape. See create_causal_mask or
        create_bidirectional_mask for shape details.

    Examples
    --------
    >>> # Causal mask
    >>> mask = create_attention_mask(seq_len=10, is_causal=True, batch_size=2, num_heads=8)
    >>> mask.shape
    torch.Size([2, 8, 10, 10])

    >>> # Bidirectional mask with padding
    >>> padding = torch.tensor([[1, 1, 1, 0, 0]])
    >>> mask = create_attention_mask(seq_len=5, is_causal=False, padding_mask=padding)
    >>> mask.shape
    torch.Size([1, 5, 5])
    """
    if is_causal:
        return create_causal_mask(
            seq_len=seq_len,
            batch_size=batch_size,
            num_heads=num_heads,
            device=device,
            dtype=dtype,
        )
    else:
        return create_bidirectional_mask(
            seq_len=seq_len,
            padding_mask=padding_mask,
            batch_size=batch_size,
            num_heads=num_heads,
            device=device,
            dtype=dtype,
        )
