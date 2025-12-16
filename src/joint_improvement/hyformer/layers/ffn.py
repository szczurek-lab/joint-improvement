# mlp.py
import torch
from torch import nn

MLP_RATIO = 4
MULTIPLE_OF = 256


class SwiGLUFFN(nn.Module):
    """LLaMA-style SwiGLU FFN."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden_dim = int((2 / 3) * MLP_RATIO * d_model)
        hidden_dim = (hidden_dim + MULTIPLE_OF - 1) // MULTIPLE_OF * MULTIPLE_OF

        self.w1 = nn.Linear(d_model, hidden_dim, bias=True)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=True)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(self.swiglu(x) * self.w2(x))
