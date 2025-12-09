# rmsnorm.py
import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Implements RMSNorm as used in LLaMA models. Normalizes inputs by their
    root mean square and scales with a learned weight parameter.

    Parameters
    ----------
    dim : int
        Dimension of the input features to normalize.
    eps : float, default=1e-6
        Small epsilon value added to the denominator for numerical stability.

    Attributes
    ----------
    eps : float
        Epsilon value for numerical stability.
    weight : nn.Parameter
        Learnable scaling parameter of shape [dim].

    Notes
    -----
    RMSNorm normalizes by:
        x_norm = x * rsqrt(mean(x^2) + eps)
        output = weight * x_norm

    This is more efficient than LayerNorm as it doesn't subtract the mean,
    only normalizes by the RMS.

    References
    ----------
    .. [1] Zhang, B. & Sennrich, R. (2019). "Root Mean Square Layer Normalization".
           Advances in Neural Information Processing Systems 32.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, T, D] where B is batch size,
            T is sequence length, and D is the feature dimension.

        Returns
        -------
        torch.Tensor
            Normalized and scaled tensor of shape [B, T, D].
        """
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(norm + self.eps)
        return self.weight * x_norm
