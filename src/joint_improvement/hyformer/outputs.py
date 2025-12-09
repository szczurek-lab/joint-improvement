"""Output containers for backbone models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass
class ModelOutput:
    """Base output for backbone models."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    extras: dict[str, Any] | None = None


__all__ = ["ModelOutput"]
