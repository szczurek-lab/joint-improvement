# hyformer/models/layers/rotary.py

import torch
import torch.nn as nn

ROTARY_BASE = 10000
ROTARY_DTYPE = torch.float32


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) for transformer attention.

    Applies rotary positional encoding to query and key tensors, enabling
    relative positional information to be encoded in the attention mechanism.
    The offset parameter allows applying rotary encoding to sequences incrementally.

    Parameters
    ----------
    head_dim : int
        Dimension of each attention head. Must be even.

    Attributes
    ----------
    hidden_dim : int
        Dimension of each attention head (same as head_dim).
    thetas : torch.Tensor
        Precomputed frequency basis tensors for rotary encoding, shape
        [1, 1, 1, num_features, 1]. Registered as a non-persistent buffer.

    Raises
    ------
    AssertionError
        If head_dim is not even.

    Notes
    -----
    The input tensor should have shape [BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM].
    The offset parameter allows applying rotary encoding to sequences that
    are part of a larger sequence, by specifying how many tokens precede
    the current input.

    References
    ----------
    .. [1] Su, J. et al. (2021). "RoFormer: Enhanced Transformer with Rotary
           Position Embedding". arXiv preprint arXiv:2104.09864.
    """

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        assert head_dim % 2 == 0
        self.hidden_dim = head_dim
        num_features = self.hidden_dim // 2
        thetas = ROTARY_BASE ** (-torch.arange(0, num_features, dtype=ROTARY_DTYPE) / num_features).reshape(
            1, 1, 1, num_features, 1
        )
        self.register_buffer("thetas", thetas, persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Apply rotary positional encoding to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM].
        offset : int, default=0
            Number of tokens that precede the input in the sequence.
            Used for incremental encoding of sequence parts.

        Returns
        -------
        torch.Tensor
            Tensor with rotary positional encoding applied, same shape as input.

        Raises
        ------
        AssertionError
            If input tensor doesn't have 4 dimensions or offset is negative.
        """
        assert len(x.shape) == 4
        assert offset >= 0
        batch_size, seq_len, num_head, head_dim = x.shape
        device = x.device

        ms = torch.arange(offset, offset + seq_len, device=device).reshape(1, seq_len, 1, 1, 1)
        angles = ms * self.thetas
        cosines = torch.cos(angles).to(ROTARY_DTYPE)
        sines = torch.sin(angles).to(ROTARY_DTYPE)

        x_grp = x.reshape(batch_size, seq_len, num_head, head_dim // 2, 2)
        x_cos = x_grp * cosines
        x_sin = x_grp * sines

        result = x_cos + torch.stack([-x_sin[..., 1], x_sin[..., 0]], dim=-1)
        result = result.flatten(-2)
        return result
