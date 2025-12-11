# mlp.py
import torch
import torch.nn.functional as F
from torch import nn

MLP_RATIO = 4.0


class SwiGLUMLP(nn.Module):
    """
    LLaMA-style SwiGLU MLP (Swish-Gated Linear Unit).

    Implements a gated MLP with SiLU activation, following the architecture
    used in LLaMA models. The SwiGLU activation combines a gated projection
    with SiLU activation and an up projection.

    Parameters
    ----------
    d_model : int
        Input and output dimension of the model.

    Attributes
    ----------
    gate_proj : nn.Linear
        Linear projection for the gate, shape [d_model, hidden_dim].
    up_proj : nn.Linear
        Linear projection for the up branch, shape [d_model, hidden_dim].
    down_proj : nn.Linear
        Linear projection for the output, shape [hidden_dim, d_model].

    Notes
    -----
    The forward pass computes:
        hidden = silu(W_gate x) * (W_up x)
        out = W_down hidden

    References
    ----------
    .. [1] Shazeer, N. (2020). "GLU Variants Improve Transformer".
           arXiv preprint arXiv:2002.05202.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden_dim = int(MLP_RATIO * d_model)

        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=True)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=True)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SwiGLU MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, T, d_model] where B is batch size,
            T is sequence length, and d_model is the model dimension.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [B, T, d_model].
        """
        gate = F.silu(self.gate_proj(x))  # SiLU = "Swish"
        up = self.up_proj(x)
        hidden = gate * up
        return self.down_proj(hidden)
