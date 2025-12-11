# kv_cache.py

import torch


class KVCache:
    """
    Head-major KV cache for autoregressive decoding.

    Implements a key-value cache for efficient autoregressive generation in
    transformer models. Supports two modes: preallocated (faster, fixed size)
    and dynamic (flexible, uses concatenation).

    Parameters
    ----------
    batch_size : int
        Batch size for the cached tensors.
    num_heads : int
        Number of attention heads.
    head_dim : int
        Dimension of each attention head.
    max_seq_len : Optional[int], default=None
        Maximum sequence length for preallocated mode. If None, uses dynamic
        mode with concatenation.
    dtype : torch.dtype, default=torch.float16
        Data type for the cache tensors.
    device : torch.device | str, default="cuda"
        Device to store the cache tensors on.

    Attributes
    ----------
    B : int
        Batch size.
    H : int
        Number of attention heads.
    Hd : int
        Head dimension.
    max_seq_len : Optional[int]
        Maximum sequence length (None for dynamic mode).
    device : torch.device
        Device where tensors are stored.
    dtype : torch.dtype
        Data type of cached tensors.
    k : torch.Tensor | None
        Cached key tensor of shape [B, H, T, Hd] (or None if empty in dynamic mode).
    v : torch.Tensor | None
        Cached value tensor of shape [B, H, T, Hd] (or None if empty in dynamic mode).
    length : int
        Current number of cached tokens.

    Notes
    -----
    Layout: K, V tensors have shape [B, H, T, Hd] where:
        - B: batch size
        - H: number of heads
        - T: sequence length (varies)
        - Hd: head dimension

    Two modes:
        - Preallocated (max_seq_len provided): Fast, no concatenation needed
        - Dynamic (max_seq_len=None): Flexible, uses torch.cat for appending
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int | None = None,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cuda",
    ) -> None:
        self.B = batch_size
        self.H = num_heads
        self.Hd = head_dim
        self.max_seq_len = max_seq_len
        self.device = torch.device(device)
        self.dtype = dtype

        self._prealloc = max_seq_len is not None

        if self._prealloc:
            self.k = torch.zeros(
                (self.B, self.H, max_seq_len, self.Hd),
                dtype=dtype,
                device=self.device,
            )
            self.v = torch.zeros_like(self.k)
            self.length = 0
        else:
            self.k = None
            self.v = None
            self.length = 0

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """
        Append new key-value pairs to the cache.

        Parameters
        ----------
        k_new : torch.Tensor
            New key tensor of shape [B, H, T_new, Hd].
        v_new : torch.Tensor
            New value tensor of shape [B, H, T_new, Hd].

        Raises
        ------
        AssertionError
            If batch size, number of heads, or head dimension don't match,
            or if cache would overflow in preallocated mode.
        """
        B, H, T_new, Hd = k_new.shape
        assert B == self.B and H == self.H and Hd == self.Hd

        if self._prealloc:
            assert self.length + T_new <= self.max_seq_len, "KVCache overflow"
            self.k[:, :, self.length : self.length + T_new, :] = k_new
            self.v[:, :, self.length : self.length + T_new, :] = v_new
            self.length += T_new
        else:
            if self.k is None:
                self.k = k_new
                self.v = v_new
            else:
                self.k = torch.cat([self.k, k_new], dim=2)
                self.v = torch.cat([self.v, v_new], dim=2)
            self.length = self.k.shape[2]

    def get_kv(self, upto: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve cached key-value pairs up to a specified position.

        Parameters
        ----------
        upto : Optional[int], default=None
            Number of tokens to retrieve. If None, returns all cached tokens.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (k, v) tensors, each of shape [B, H, upto, Hd].
        """
        if upto is None:
            upto = self.length
        return (
            self.k[:, :, :upto, :],
            self.v[:, :, :upto, :],
        )

    def is_empty(self) -> bool:
        """
        Check if the cache is empty.

        Returns
        -------
        bool
            True if no tokens are cached, False otherwise.
        """
        return self.length == 0

    def reset(self) -> None:
        """
        Reset the cache to empty state.

        In preallocated mode, only resets the length counter.
        In dynamic mode, clears the k and v tensors.
        """
        if self._prealloc:
            self.length = 0
        else:
            self.k = None
            self.v = None
            self.length = 0
