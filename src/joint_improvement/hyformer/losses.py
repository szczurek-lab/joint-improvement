"""Loss functions for task training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    shift_labels: bool = True,
) -> torch.Tensor:
    """Compute language modeling loss."""
    logits = logits.contiguous()
    labels = labels.contiguous()

    if shift_labels:
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

    B, T = labels.shape
    V = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(B * T, V),
        labels.reshape(B * T),
        reduction="mean",
    )


def compute_mlm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute masked language modeling loss with optional label smoothing.

    The loss is computed only over masked positions (where labels != -100).
    Non-masked positions are ignored in the loss computation.
    """
    # Make contiguous at the loss boundary (torch.compile-friendly)
    logits = logits.contiguous()
    labels = labels.contiguous()

    batch_size, seq_len = logits.shape[:2]
    vocab_size = logits.size(-1)

    logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
    labels_flat = labels.reshape(batch_size * seq_len)

    # Create mask for valid (masked) positions
    valid_mask = labels_flat != -100

    if not valid_mask.any():
        # No masked positions, return zero loss
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)

    if label_smoothing > 0.0:
        # Create one-hot labels only for valid positions
        one_hot = torch.zeros_like(logits_flat)
        valid_labels = labels_flat[valid_mask]
        valid_indices = torch.arange(len(labels_flat), device=labels_flat.device)[valid_mask]
        one_hot[valid_indices].scatter_(1, valid_labels.unsqueeze(1), 1.0)

        # Smooth labels: (1 - smoothing) * one_hot + smoothing / vocab_size
        smooth_labels = (1.0 - label_smoothing) * one_hot + label_smoothing / vocab_size

        # Compute cross-entropy loss with smoothed labels
        # log_softmax + nll_loss = cross_entropy
        log_probs = torch.nn.functional.log_softmax(logits_flat, dim=-1)
        # Only compute loss for valid positions
        loss = -torch.sum(smooth_labels * log_probs, dim=-1)
        loss = loss[valid_mask].mean()
    else:
        # Standard cross-entropy loss (no smoothing)
        # Explicitly set ignore_index=-100 for MLM (default but explicit helps Inductor)
        # F.cross_entropy with ignore_index automatically normalizes over non-ignored positions
        loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100, reduction="mean")

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
    # Expected shapes: logits [B, C], labels [B]
    if labels.ndim != 1:
        raise ValueError(f"Single-label classification expects labels shape [B], got {labels.shape}")

    logits = logits.contiguous()
    labels_clean = labels.contiguous().clone()

    # Only floating labels can contain NaNs
    if labels_clean.is_floating_point():
        nan_mask = torch.isnan(labels_clean)
        if nan_mask.any():
            labels_clean[nan_mask] = ignore_index

    # If reduction is mean, we need to account for missing values
    if reduction == "mean":
        # Count valid (non-missing) samples
        if labels.is_floating_point():
            valid_mask = (labels_clean != ignore_index) & (~torch.isnan(labels))
        else:
            valid_mask = labels_clean != ignore_index
        num_valid = valid_mask.sum().float()
        if num_valid > 0:
            # Recompute with proper normalization
            loss_sum = F.cross_entropy(logits, labels_clean.long(), ignore_index=ignore_index, reduction="sum")
            loss = loss_sum / num_valid
        else:
            # All samples are missing, return zero loss
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
    else:
        # Use ignore_index to mask out missing values
        loss = F.cross_entropy(logits, labels_clean.long(), ignore_index=ignore_index, reduction=reduction)
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
    loss_per_label = F.binary_cross_entropy_with_logits(logits, labels_float, reduction="none")  # [B, num_labels]

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


def compute_multitarget_regression_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute multi-target regression loss with support for missing values.

    This is MSE over all valid (non-NaN) target entries.

    Parameters
    ----------
    logits : torch.Tensor
        Predictions tensor of shape [B, D].
    labels : torch.Tensor
        Target tensor of shape [B, D]. Missing values should be NaN.
    reduction : str, default="mean"
        Reduction method: "mean", "sum", or "none".
    """
    if logits.ndim != 2 or labels.ndim != 2:
        raise ValueError(f"Expected logits/labels to be 2D [B, D]. Got logits={logits.shape}, labels={labels.shape}")

    logits = logits.contiguous()
    labels_float = labels.contiguous().to(dtype=logits.dtype)
    valid_mask = ~torch.isnan(labels_float)

    squared_errors = (logits - labels_float) ** 2

    if reduction == "none":
        out = squared_errors
        out[~valid_mask] = float("nan")
        return out

    # Aggregate only valid entries
    num_valid = valid_mask.sum().to(dtype=logits.dtype)
    if num_valid == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    loss_sum = torch.where(valid_mask, squared_errors, torch.zeros_like(squared_errors)).sum()
    if reduction == "sum":
        return loss_sum
    if reduction == "mean":
        return loss_sum / num_valid
    raise ValueError(f"Unsupported reduction: {reduction}. Choose from 'mean', 'sum', 'none'")


def compute_prediction_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -1,
    reduction: str = "mean",
    task_type: str,
) -> torch.Tensor:
    """
    Compute prediction loss for downstream tasks (classification or regression).

    Select the appropriate prediction loss based on `task_type`.

    This function is intentionally strict: loss selection should come from the
    configured prediction task type (e.g. via `HyformerConfig.prediction_task_type`)
    rather than heuristics on dtype/shape.

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
    ignore_index : int, default=-1
        Value to ignore in labels for single-label classification (treated as missing).
        Also ignores NaN values.
    reduction : str, default="mean"
        Reduction method: "mean", "sum", or "none".
    task_type : str
        Prediction task type to select the loss function.
        Supported values:
        - "multitarget_regression"
        - "regression"
        - "classification"
        - "multilabel_classification"

    Returns
    -------
    torch.Tensor
        Loss tensor (scalar if reduction="mean" or "sum", tensor if reduction="none").

    Examples
    --------
    >>> # Multi-class classification (single-label) with missing values
    >>> logits = torch.randn(4, 10)
    >>> labels = torch.tensor([0, 1, -1, 5])  # -1 is missing
    >>> loss = compute_prediction_loss(logits, labels, task_type="classification")
    >>> loss.shape
    torch.Size([])  # Scalar

    >>> # Binary classification (single-label) with missing values
    >>> logits = torch.randn(4, 2)
    >>> labels = torch.tensor([0, 1, -1, 1])  # -1 is missing
    >>> loss = compute_prediction_loss(logits, labels, task_type="classification")
    >>> loss.shape
    torch.Size([])  # Scalar

    >>> # Multi-label classification
    >>> logits = torch.randn(4, 5)  # 5 classes
    >>> labels = torch.tensor([[1, 0, 1, 0, 0], [0, 1, 0, 1, 1], [1, 1, 0, 0, 0], [0, 0, 0, 1, 0]])
    >>> loss = compute_prediction_loss(logits, labels, task_type="multilabel_classification")
    >>> loss.shape
    torch.Size([])  # Scalar

    >>> # Regression with missing values
    >>> logits = torch.randn(4, 1)
    >>> labels = torch.tensor([[1.0], [2.0], [float("nan")], [3.0]])  # NaN is missing
    >>> loss = compute_prediction_loss(logits, labels, task_type="regression")
    >>> loss.shape
    torch.Size([])  # Scalar

    Notes
    -----
    - Binary classification (single-label): num_labels=2, labels in [0, 1], uses CrossEntropyLoss
    - Multi-class classification (single-label): num_labels>2, labels in [0, num_labels-1], uses CrossEntropyLoss
    - Multi-label classification: labels shape [B, num_labels] with values 0 or 1, uses BCEWithLogitsLoss
    - Missing values are automatically handled and excluded from loss computation
    """
    if task_type == "multitarget_regression":
        return compute_multitarget_regression_loss(logits, labels, reduction=reduction)
    if task_type == "regression":
        return compute_regression_loss(logits, labels, reduction=reduction)
    if task_type == "multilabel_classification":
        return compute_multilabel_classification(logits, labels, reduction=reduction)
    if task_type == "classification":
        return compute_binary_classification(logits, labels, ignore_index=ignore_index, reduction=reduction)

    raise ValueError(
        "Unsupported task_type. Expected one of "
        "['multitarget_regression', 'regression', 'classification', 'multilabel_classification'], "
        f"got {task_type!r}"
    )
