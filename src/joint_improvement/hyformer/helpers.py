"""Helper utilities for backbone loading and inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .inputs import ModelInput


def load_checkpoint(path_like: str | Path) -> tuple[dict[str, Any], Any | None]:
    """Load state dict (and optional config) from a local directory or file."""
    path = Path(path_like)
    config = None

    if path.is_dir():
        state_file = _find_first_existing(
            path,
            (
                "pytorch_model.bin",
                "model.safetensors",
                "model.pt",
                "model.pth",
            ),
        )
        if state_file is None:
            raise FileNotFoundError(f"No state dict found in directory {path}")
        state_dict = _load_state_dict(state_file)
        config_file = _find_first_existing(path, ("config.pt", "config.pth"))
        if config_file is not None:
            config = torch.load(config_file, map_location="cpu")
        return state_dict, config

    if path.suffix in {".bin", ".pt", ".pth", ".safetensors"}:
        return _load_state_dict(path), None

    raise FileNotFoundError(f"Unsupported checkpoint format or path not found: {path}")


def coerce_to_inputs(
    inputs: ModelInput | None = None,
    *,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    targets: torch.Tensor | None = None,
    task: str = "",
) -> ModelInput:
    """Convert loose tensors or kwargs into a `ModelInput` instance.

    Parameters
    ----------
    inputs : ModelInput | None
        Existing ModelInput instance to return directly.
    input_ids : torch.Tensor | None
        Input token IDs tensor.
    attention_mask : torch.Tensor | None
        Attention mask tensor.
    labels : torch.Tensor | None
        Labels tensor (for sequence-level tasks).
    targets : torch.Tensor | None
        Targets tensor (for prediction tasks).
    task : str
        Task name/identifier.

    Returns
    -------
    ModelInput
        ModelInput instance constructed from the provided tensors.
    """
    if inputs is not None:
        return inputs

    if input_ids is None:
        raise ValueError("Expected `inputs` or `input_ids` to construct ModelInput.")

    return ModelInput(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        targets=targets,
        task=task,
    )


def _find_first_existing(directory: Path, candidates: tuple[str, ...]) -> Path | None:
    """Return the first existing file from candidates in the directory."""
    return next((directory / name for name in candidates if (directory / name).exists()), None)


def _load_state_dict(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                f"Loading {path} requires safetensors. Install it or provide .pt/.bin checkpoints."
            ) from exc
        return load_file(path)
    return torch.load(path, map_location="cpu")


__all__ = ["load_checkpoint", "coerce_to_inputs"]
