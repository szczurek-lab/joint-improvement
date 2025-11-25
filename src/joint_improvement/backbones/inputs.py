"""Input container for backbone models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class ModelInput:
    """Input container for backbone models."""

    input_ids: torch.Tensor
    task: str
    attention_mask: torch.Tensor | None = None
    labels: torch.Tensor | None = None


__all__ = ["ModelInput"]
