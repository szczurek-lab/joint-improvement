"""Prediction head for Hyformer model (regression and classification tasks)."""

from __future__ import annotations

import torch
from torch import nn

from joint_improvement.hyformer.losses import compute_prediction_loss
from joint_improvement.hyformer.model import Hyformer


class PredictionHead(nn.Module):
    """
    Prediction head for downstream tasks (classification or regression).

    Wraps the base Hyformer model with a task-specific prediction head.
    Supports binary classification, multi-class classification, and regression.
    Handles missing values in labels.

    Parameters
    ----------
    base_model : Hyformer
        Base transformer model to wrap.
    num_labels : int
        Number of output labels/classes for the prediction task.
        - For binary classification: num_labels=2
        - For multi-class classification: num_labels>2
        - For regression: num_labels=1
    d_model : int | None, default=None
        Model dimension (hidden size). If None, inferred from base_model.
    dropout : float, default=0.1
        Dropout probability applied before the classifier.
    ignore_index : int, default=-1
        Value to ignore in labels for classification (treated as missing).
        Also ignores NaN values.

    Attributes
    ----------
    base_model : Hyformer
        The base transformer model.
    num_labels : int
        Number of output labels.
    norm : nn.LayerNorm
        Layer normalization applied to CLS token (DINOv2 best practice).
    dropout : nn.Dropout
        Dropout layer.
    classifier : nn.Linear
        Classification/regression head.
    ignore_index : int
        Value used to mark missing labels.

    Notes
    -----
    This head uses the CLS token (first token) representation and projects
    to the number of labels.

    Missing values handling:
    - For classification: Missing values can be represented as NaN or ignore_index.
      Only valid samples contribute to the loss.
    - For regression: Missing values should be represented as NaN.
      Only valid samples contribute to the loss.

    Examples
    --------
    >>> from ..model import Hyformer
    >>> from ..config import HyformerConfig
    >>> config = HyformerConfig(vocab_size=32000, d_model=512, n_heads=8)
    >>> base_model = Hyformer.from_config(config)
    >>> # For binary classification
    >>> model = PredictionHead(base_model, num_labels=2, d_model=512)
    >>> input_ids = torch.randint(0, 32000, (4, 10))
    >>> attention_mask = torch.ones(4, 10)
    >>> labels = torch.tensor([0, 1, -1, 1])  # -1 is missing
    >>> logits, loss = model(input_ids, attention_mask=attention_mask, labels=labels, task="prediction")
    >>> logits.shape
    torch.Size([4, 2])

    >>> # For regression
    >>> model = PredictionHead(base_model, num_labels=1, d_model=512)
    >>> labels = torch.tensor([[1.0], [2.0], [float("nan")], [3.0]])  # NaN is missing
    >>> logits, loss = model(input_ids, attention_mask=attention_mask, labels=labels, task="prediction")
    >>> logits.shape
    torch.Size([4, 1])
    """

    def __init__(
        self,
        base_model: Hyformer,
        num_labels: int,
        d_model: int | None = None,
        dropout: float = 0.1,
        ignore_index: int = -1,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        # Get d_model from base model if not provided
        if d_model is None:
            d_model = base_model.embed.embedding_dim
        self.d_model = d_model
        self.ignore_index = ignore_index

        # DINOv2 best practices: LayerNorm before classifier
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        task: str,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for prediction task.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs of shape [B, T] where B is batch size and
            T is sequence length.
        task : str
            Task name (should be "prediction" for this head).
        attention_mask : Optional[torch.Tensor], default=None
            Attention mask tensor of shape [B, T] where 1 indicates valid
            tokens and 0 indicates padding.
        labels : Optional[torch.Tensor], default=None
            Target labels of shape [B] for classification or [B, num_labels] or [B]
            for regression. Missing values: NaN or ignore_index for classification,
            NaN for regression. If provided, loss will be computed and returned.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            If labels is None:
                - Logits tensor of shape [B, num_labels]
            If labels is provided:
                - Logits tensor of shape [B, num_labels]
                - Loss tensor (scalar)

        Notes
        -----
        Uses the CLS token (first token) representation for prediction.
        Follows DINOv2 best practices:
        - Extracts CLS token directly
        - Applies LayerNorm before classifier for stable training
        - Uses dropout for regularization
        """
        # Prediction uses bidirectional attention
        outputs = self.base_model(
            input_ids=input_ids,
            task=task,
            attention_mask=attention_mask,
            kv_caches=None,
            use_cache=False,
        )

        # Get hidden states from model output extras
        hidden_states = outputs.extras["hidden_states"]  # [B, T, d_model]

        # Use CLS token (first token) - following DINOv2 best practices
        pooled = hidden_states[:, 0, :]  # [B, d_model]

        # DINOv2 best practices: Normalize before classifier for stable training
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        if labels is not None:
            # Compute prediction loss (classification or regression) with missing value support
            loss = compute_prediction_loss(
                logits,
                labels,
                num_labels=self.num_labels,
                ignore_index=self.ignore_index,
                reduction="mean",
            )
            return logits, loss

        return logits
