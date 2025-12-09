"""Loss functions for task training."""

from __future__ import annotations

import torch
from torch import nn


def compute_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    shift_labels: bool = True,
) -> torch.Tensor:
    """
    Compute language modeling (causal LM) loss.

    Computes cross-entropy loss for next-token prediction. Optionally shifts
    labels to align with predictions (standard for causal language modeling).

    Parameters
    ----------
    logits : torch.Tensor
        Logits tensor of shape [B, T, vocab_size] where B is batch size,
        T is sequence length, and vocab_size is vocabulary size.
    labels : torch.Tensor
        Target token IDs of shape [B, T].
    shift_labels : bool, default=True
        Whether to shift labels for causal LM. If True, shifts labels by one
        position so that position i predicts position i+1.

    Returns
    -------
    torch.Tensor
        Loss tensor (scalar).

    Examples
    --------
    >>> logits = torch.randn(2, 10, 32000)
    >>> labels = torch.randint(0, 32000, (2, 10))
    >>> loss = compute_lm_loss(logits, labels)
    >>> loss.shape
    torch.Size([])  # Scalar
    """
    if shift_labels:
        # Shift so that tokens < n predict n (standard for causal LM)
        shift_logits = logits[..., :-1, :].contiguous()
        target_labels = labels[..., 1:].contiguous()
    else:
        shift_logits = logits
        target_labels = labels

    # Flatten the tokens
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), target_labels.view(-1))

    return loss


def compute_mlm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Compute masked language modeling (MLM) loss with optional label smoothing.

    Computes cross-entropy loss for masked token prediction. Supports label
    smoothing for regularization (BERT-style).

    Parameters
    ----------
    logits : torch.Tensor
        Logits tensor of shape [B, num_masked, vocab_size] or [B*num_masked, vocab_size]
        where B is batch size and num_masked is number of masked positions.
    labels : torch.Tensor
        Target token IDs of shape [B, num_masked] or [B*num_masked].
    vocab_size : int
        Size of the vocabulary.
    label_smoothing : float, default=0.0
        Label smoothing factor. Smooths the one-hot labels by mixing with
        uniform distribution. 0.0 means no smoothing (hard labels).
        Typical values: 0.0-0.2.

    Returns
    -------
    torch.Tensor
        Loss tensor (scalar).

    Examples
    --------
    >>> logits = torch.randn(2, 5, 32000)  # 5 masked positions
    >>> labels = torch.randint(0, 32000, (2, 5))
    >>> loss = compute_mlm_loss(logits, labels, vocab_size=32000, label_smoothing=0.1)
    >>> loss.shape
    torch.Size([])  # Scalar

    Notes
    -----
    Label smoothing formula:
        smooth_labels = (1 - smoothing) * one_hot + smoothing / vocab_size

    This acts as regularization and prevents overconfident predictions.
    """
    # Flatten for loss computation
    logits_flat = logits.view(-1, vocab_size)  # [B*num_masked, vocab_size]
    labels_flat = labels.view(-1)  # [B*num_masked]

    if label_smoothing > 0.0:
        # Label smoothing: mix one-hot labels with uniform distribution
        # Create one-hot labels
        one_hot = torch.zeros_like(logits_flat)
        one_hot.scatter_(1, labels_flat.unsqueeze(1), 1.0)

        # Smooth labels: (1 - smoothing) * one_hot + smoothing / vocab_size
        smooth_labels = (1.0 - label_smoothing) * one_hot + label_smoothing / vocab_size

        # Compute cross-entropy loss with smoothed labels
        # log_softmax + nll_loss = cross_entropy
        log_probs = torch.nn.functional.log_softmax(logits_flat, dim=-1)
        loss = -torch.sum(smooth_labels * log_probs, dim=-1).mean()
    else:
        # Standard cross-entropy loss (no smoothing)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits_flat, labels_flat)

    return loss


def compute_binary_classification(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -1,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute single-label classification loss (binary or multi-class).

    Uses CrossEntropyLoss for classification tasks where each sample belongs
    to exactly one class. Supports missing values represented as NaN or ignore_index.

    Parameters
    ----------
    logits : torch.Tensor
        Logits tensor of shape [B, num_labels] where B is batch size and
        num_labels is number of classes. For binary classification, num_labels=2.
    labels : torch.Tensor
        Target class indices of shape [B] with values in [0, num_labels-1].
        Missing values can be represented as NaN or ignore_index.
    ignore_index : int, default=-1
        Value to ignore in labels (treated as missing). Also ignores NaN values.
    reduction : str, default="mean"
        Reduction method: "mean", "sum", or "none".

    Returns
    -------
    torch.Tensor
        Loss tensor (scalar if reduction="mean" or "sum", tensor if reduction="none").

    Examples
    --------
    >>> # Multi-class classification
    >>> logits = torch.randn(4, 10)  # 10 classes
    >>> labels = torch.tensor([0, 1, -1, 5])  # -1 is missing
    >>> loss = compute_binary_classification(logits, labels)
    >>> loss.shape
    torch.Size([])  # Scalar

    >>> # Binary classification
    >>> logits = torch.randn(4, 2)  # 2 classes
    >>> labels = torch.tensor([0, 1, -1, 1])  # -1 is missing
    >>> loss = compute_binary_classification(logits, labels)
    >>> loss.shape
    torch.Size([])  # Scalar

    """
    # Single-label classification: Use CrossEntropyLoss (softmax over classes)
    # labels: [B] with class indices
    labels_clean = labels.clone()
    if torch.isnan(labels).any():
        labels_clean[torch.isnan(labels)] = ignore_index

    # Use ignore_index to mask out missing values
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
    loss = loss_fct(logits, labels_clean.long())

    # If reduction is mean, we need to account for missing values
    if reduction == "mean":
        # Count valid (non-missing) samples
        valid_mask = (labels_clean != ignore_index) & (~torch.isnan(labels))
        num_valid = valid_mask.sum().float()
        if num_valid > 0:
            # Recompute with proper normalization
            loss_fct_sum = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="sum")
            loss_sum = loss_fct_sum(logits, labels_clean.long())
            loss = loss_sum / num_valid
        else:
            # All samples are missing, return zero loss
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

    return loss


def compute_multilabel_classification(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute multi-label classification loss.

    Uses BCEWithLogitsLoss for classification tasks where each sample can belong
    to multiple classes simultaneously. Supports missing values represented as NaN.

    Parameters
    ----------
    logits : torch.Tensor
        Logits tensor of shape [B, num_labels] where B is batch size and
        num_labels is number of classes.
    labels : torch.Tensor
        Binary labels of shape [B, num_labels] with values 0 or 1.
        Each sample can have multiple active labels.
        Missing values can be represented as NaN.
    reduction : str, default="mean"
        Reduction method: "mean", "sum", or "none".

    Returns
    -------
    torch.Tensor
        Loss tensor (scalar if reduction="mean" or "sum", tensor if reduction="none").

    Examples
    --------
    >>> # Multi-label classification
    >>> logits = torch.randn(4, 5)  # 5 classes
    >>> labels = torch.tensor(
    ...     [
    ...         [1, 0, 1, 0, 0],  # Sample 1: classes 0 and 2
    ...         [0, 1, 0, 1, 1],  # Sample 2: classes 1, 3, and 4
    ...         [1, 1, 0, 0, 0],  # Sample 3: classes 0 and 1
    ...         [0, 0, 0, 1, 0],  # Sample 4: only class 3
    ...     ]
    ... )
    >>> loss = compute_multilabel_classification(logits, labels)
    >>> loss.shape
    torch.Size([])  # Scalar

    Notes
    -----
    - Uses BCEWithLogitsLoss (sigmoid per class, multiple labels per sample)
    - Each class is predicted independently
    - Missing values (NaN) are masked out per-label
    """
    # Multi-label classification: Use BCE with logits (sigmoid per class)
    # labels: [B, num_labels] with values 0 or 1
    labels_float = labels.float()

    # Mask out NaN values
    if torch.isnan(labels_float).any():
        # Create mask for valid labels (not NaN)
        valid_mask = ~torch.isnan(labels_float)
        labels_float = torch.where(valid_mask, labels_float, torch.zeros_like(labels_float))
    else:
        valid_mask = torch.ones_like(labels_float, dtype=torch.bool)

    # Use BCEWithLogitsLoss
    loss_fct = nn.BCEWithLogitsLoss(reduction="none")  # No reduction for masking
    loss_per_label = loss_fct(logits, labels_float)  # [B, num_labels]

    # Apply mask to ignore NaN values
    if torch.isnan(labels).any():
        loss_per_label = torch.where(valid_mask, loss_per_label, torch.zeros_like(loss_per_label))

    if reduction == "mean":
        # Average over all valid label predictions
        num_valid = valid_mask.sum().float()
        if num_valid > 0:
            loss = loss_per_label.sum() / num_valid
        else:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
    elif reduction == "sum":
        loss = loss_per_label.sum()
    else:  # "none"
        loss = loss_per_label

    return loss


def compute_regression_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute regression loss with support for missing values.

    Computes mean squared error (MSE) loss for regression tasks.
    Supports missing values represented as NaN.

    Parameters
    ----------
    logits : torch.Tensor
        Predictions tensor of shape [B, 1] or [B] where B is batch size.
    labels : torch.Tensor
        Target values of shape [B, 1] or [B]. Missing values should be NaN.
    reduction : str, default="mean"
        Reduction method: "mean", "sum", or "none".

    Returns
    -------
    torch.Tensor
        Loss tensor (scalar if reduction="mean" or "sum", tensor if reduction="none").

    Examples
    --------
    >>> logits = torch.randn(4, 1)  # Regression predictions
    >>> labels = torch.tensor([[1.0], [2.0], [float("nan")], [3.0]])  # NaN is missing
    >>> loss = compute_regression_loss(logits, labels)
    >>> loss.shape
    torch.Size([])  # Scalar

    Notes
    -----
    Missing values (NaN) are automatically masked out when computing the loss.
    Only valid (non-NaN) samples contribute to the loss.
    """
    logits_flat = logits.squeeze()
    labels_flat = labels.squeeze().float()

    # Mask out NaN values
    valid_mask = ~torch.isnan(labels_flat)
    num_valid = valid_mask.sum().float()

    if num_valid == 0:
        # All samples are missing, return zero loss
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Compute squared errors for all samples
    squared_errors = (logits_flat - labels_flat) ** 2

    if reduction == "mean":
        # Average over valid samples only
        loss = squared_errors[valid_mask].mean()
    elif reduction == "sum":
        # Sum over valid samples only
        loss = squared_errors[valid_mask].sum()
    elif reduction == "none":
        # Return full tensor with NaN for missing values
        loss = squared_errors.clone()
        loss[~valid_mask] = float("nan")
    else:
        raise ValueError(f"Unsupported reduction: {reduction}. Choose from 'mean', 'sum', 'none'")

    return loss


def compute_prediction_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_labels: int,
    ignore_index: int = -1,
    reduction: str = "mean",
    multilabel: bool = False,
) -> torch.Tensor:
    """
    Compute prediction loss for downstream tasks (classification or regression).

    Automatically selects the appropriate loss function based on num_labels.
    Supports missing values in labels.

    Parameters
    ----------
    logits : torch.Tensor
        Logits/predictions tensor of shape [B, num_labels] or [B, 1] or [B].
        For binary classification, num_labels=2.
    labels : torch.Tensor
        For single-label classification: Target labels of shape [B] with class indices.
        For multi-label classification: Target labels of shape [B, num_labels] with binary values.
        For regression: Target values of shape [B, 1] or [B].
        Missing values: NaN or ignore_index for classification, NaN for regression.
    num_labels : int
        Number of output labels. If 1, uses regression loss (MSE).
        Otherwise, uses classification loss.
        For binary classification, num_labels=2.
    ignore_index : int, default=-1
        Value to ignore in labels for single-label classification (treated as missing).
        Also ignores NaN values.
    reduction : str, default="mean"
        Reduction method: "mean", "sum", or "none".
    multilabel : bool, default=False
        If True, treats as multi-label classification (each sample can have multiple labels).
        If False, treats as single-label classification (each sample has one label).
        Only applies when num_labels > 1.

    Returns
    -------
    torch.Tensor
        Loss tensor (scalar if reduction="mean" or "sum", tensor if reduction="none").

    Examples
    --------
    >>> # Multi-class classification (single-label) with missing values
    >>> logits = torch.randn(4, 10)
    >>> labels = torch.tensor([0, 1, -1, 5])  # -1 is missing
    >>> loss = compute_prediction_loss(logits, labels, num_labels=10)
    >>> loss.shape
    torch.Size([])  # Scalar

    >>> # Binary classification (single-label) with missing values
    >>> logits = torch.randn(4, 2)
    >>> labels = torch.tensor([0, 1, -1, 1])  # -1 is missing
    >>> loss = compute_prediction_loss(logits, labels, num_labels=2)
    >>> loss.shape
    torch.Size([])  # Scalar

    >>> # Multi-label classification
    >>> logits = torch.randn(4, 5)  # 5 classes
    >>> labels = torch.tensor([[1, 0, 1, 0, 0], [0, 1, 0, 1, 1], [1, 1, 0, 0, 0], [0, 0, 0, 1, 0]])
    >>> loss = compute_prediction_loss(logits, labels, num_labels=5, multilabel=True)
    >>> loss.shape
    torch.Size([])  # Scalar

    >>> # Regression with missing values
    >>> logits = torch.randn(4, 1)
    >>> labels = torch.tensor([[1.0], [2.0], [float("nan")], [3.0]])  # NaN is missing
    >>> loss = compute_prediction_loss(logits, labels, num_labels=1)
    >>> loss.shape
    torch.Size([])  # Scalar

    Notes
    -----
    - Binary classification (single-label): num_labels=2, labels in [0, 1], uses CrossEntropyLoss
    - Multi-class classification (single-label): num_labels>2, labels in [0, num_labels-1], uses CrossEntropyLoss
    - Multi-label classification: labels shape [B, num_labels] with values 0 or 1, uses BCEWithLogitsLoss
    - Missing values are automatically handled and excluded from loss computation
    """
    if num_labels == 1:
        # Regression task
        return compute_regression_loss(logits, labels, reduction=reduction)
    elif multilabel:
        # Multi-label classification
        return compute_multilabel_classification(logits, labels, reduction=reduction)
    else:
        # Single-label classification (binary or multi-class)
        return compute_binary_classification(logits, labels, ignore_index=ignore_index, reduction=reduction)
