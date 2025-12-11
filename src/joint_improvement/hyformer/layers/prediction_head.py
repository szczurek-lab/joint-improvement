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

    Attributes
    ----------
    norm : nn.LayerNorm
        Layer normalization applied to pooled representation (DINOv2 best practice).
    dropout : nn.Dropout
        Dropout layer.
    classifier : nn.Linear
        Classification/regression head.

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
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # DINOv2 best practices: LayerNorm before classifier
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize prediction head weights following LLaMA initialization.

        Initializes the classifier with Normal(0, 0.02) following LLaMA best practices.
        LayerNorm uses default PyTorch initialization (ones for weights, zeros for bias).
        """
        import torch.nn as nn

        # Initialize classifier weights and bias
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
        # LayerNorm is already initialized with default PyTorch initialization

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
        logits = self.classifier(pooled)
        return logits
