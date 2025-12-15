"""Prediction head module for Hyformer model (regression and classification tasks)."""

from __future__ import annotations

import torch
from torch import nn


class PredictionHeadModule(nn.Module):
    """
    Prediction head module for downstream tasks (classification or regression).

    Applies LayerNorm, dropout, and linear projection following DINOv2 best practices.
    Designed to be used as a head in the Hyformer model's heads ModuleDict.

    Parameters
    ----------
    d_model : int
        Model dimension (hidden size).
    num_labels : int
        Number of output labels/classes for the prediction task.
        - For binary classification: num_labels=2
        - For multi-class classification: num_labels>2
        - For regression: num_labels=1
    dropout : float, default=0.1
        Dropout probability applied before the classifier.
    depth : int, default=1
        Depth of the prediction head after normalization. If 1, uses a single linear layer.
        If >1, uses an MLP of (Linear -> act_fn -> Dropout) repeated (depth - 1) times,
        followed by a final Linear classifier/regressor.
    act_fn : str, default="gelu"
        Activation function used for the MLP when depth > 1.
        Supported: "gelu", "relu", "silu".

    Attributes
    ----------
    norm : nn.LayerNorm
        Layer normalization applied to pooled representation (DINOv2 best practice).
    dropout : nn.Dropout
        Dropout layer.
    classifier : nn.Linear
        Classification/regression head.
    mlp : nn.Sequential | None
        Optional MLP stack used when depth > 1.

    Notes
    -----
    Follows DINOv2 best practices:
    - Applies LayerNorm before classifier for stable training
    - Uses dropout for regularization

    This module expects pooled representations (e.g., CLS token) as input,
    not full sequence hidden states. Use this in conjunction with CLS token
    extraction in the forward pass.
    """

    def __init__(
        self,
        d_model: int,
        num_labels: int,
        dropout: float,
        depth: int = 1,
        act_fn: str = "gelu",
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        # DINOv2 best practices: LayerNorm before classifier
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.depth = depth
        self.act_fn = act_fn

        self.mlp: nn.Sequential | None = None
        if depth > 1:
            activation = self._build_activation(act_fn)
            layers: list[nn.Module] = []
            for _ in range(depth - 1):
                layers.append(nn.Linear(d_model, d_model))
                layers.append(activation)
                layers.append(nn.Dropout(dropout))
            self.mlp = nn.Sequential(*layers)

        self.classifier = nn.Linear(d_model, num_labels)

        # Initialize weights
        self._init_weights()

    @staticmethod
    def _build_activation(act_fn: str) -> nn.Module:
        key = act_fn.lower().strip()
        if key == "gelu":
            return nn.GELU()
        if key == "relu":
            return nn.ReLU()
        if key in {"silu", "swish"}:
            return nn.SiLU()
        raise ValueError(f"Unsupported act_fn {act_fn!r}. Supported: 'gelu', 'relu', 'silu'.")

    def _init_weights(self) -> None:
        """Initialize prediction head weights following LLaMA initialization.

        Initializes all Linear layers with Normal(0, 0.02) following LLaMA best practices.
        LayerNorm uses default PyTorch initialization (ones for weights, zeros for bias).
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through prediction head.

        Parameters
        ----------
        pooled : torch.Tensor
            Pooled representation tensor of shape [B, d_model] where B is batch size.
            Typically the CLS token or pooled sequence representation.

        Returns
        -------
        torch.Tensor
            Logits tensor of shape [B, num_labels].
        """
        # DINOv2 best practices: Normalize before classifier for stable training
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        if self.mlp is not None:
            pooled = self.mlp(pooled)
        logits = self.classifier(pooled)
        return logits
