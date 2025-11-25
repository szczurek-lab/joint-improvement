"""Input container for backbone models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
else:
    import torch


@dataclass
class ModelInput:
    """Input container for backbone models.

    Supports dictionary unpacking for compatibility with model forward calls:
        model(**model_input)  # Works!
    """

    input_ids: torch.Tensor
    task: str
    attention_mask: torch.Tensor | None = None
    labels: torch.Tensor | None = None

    def keys(self) -> list[str]:
        """Return keys for dictionary unpacking compatibility.

        Only includes keys that should be passed to model forward:
        - input_ids (always)
        - task (always)
        - attention_mask (only if not None)
        Labels are excluded as they're used for loss computation, not forward pass.
        """
        keys = ["input_ids", "task"]
        if self.attention_mask is not None:
            keys.append("attention_mask")
        return keys

    def __getitem__(self, key: str) -> torch.Tensor | str | None:
        """Support dictionary-style access for unpacking."""
        if key == "input_ids":
            return self.input_ids
        elif key == "task":
            return self.task
        elif key == "attention_mask":
            return self.attention_mask
        elif key == "labels":
            return self.labels
        else:
            raise KeyError(f"ModelInput has no key '{key}'")

    def __iter__(self):
        """Support iteration for dictionary unpacking."""
        return iter(self.keys())

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for dictionary compatibility."""
        return key in self.keys()

    def to_dict(self) -> dict[str, torch.Tensor | str | None]:
        """Convert to dictionary for explicit conversion if needed."""
        result: dict[str, torch.Tensor | str | None] = {
            "input_ids": self.input_ids,
            "task": self.task,
        }
        if self.attention_mask is not None:
            result["attention_mask"] = self.attention_mask
        if self.labels is not None:
            result["labels"] = self.labels
        return result

    def to(self, device: torch.device | str) -> ModelInput:
        """Move all tensors to the specified device.

        Parameters
        ----------
        device : torch.device | str
            Target device (e.g., "cuda", "cpu", torch.device("cuda:0")).

        Returns
        -------
        ModelInput
            New ModelInput instance with all tensors moved to the device.
            Non-tensor fields (like `task`) remain unchanged.

        Examples
        --------
        >>> model_input = ModelInput(input_ids=..., task="prediction", attention_mask=...)
        >>> model_input_gpu = model_input.to("cuda")
        >>> model_input_gpu = model_input.to(torch.device("cuda:0"))
        """
        return ModelInput(
            input_ids=self.input_ids.to(device),
            task=self.task,
            attention_mask=self.attention_mask.to(device) if self.attention_mask is not None else None,
            labels=self.labels.to(device) if self.labels is not None else None,
        )


__all__ = ["ModelInput"]
