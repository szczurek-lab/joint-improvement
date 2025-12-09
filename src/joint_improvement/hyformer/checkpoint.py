"""Hyformer checkpoint dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger

try:
    from huggingface_hub import hf_hub_download

    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    hf_hub_download = None

try:
    from safetensors.torch import load_file as safetensors_load
    from safetensors.torch import save_file as safetensors_save

    _SAFETENSORS_AVAILABLE = True
except ImportError:
    _SAFETENSORS_AVAILABLE = False
    safetensors_load = None
    safetensors_save = None


@dataclass
class HyformerCheckpoint:
    """Hyformer checkpoint containing model state dictionary."""

    state_dict: dict[str, Any]

    @staticmethod
    def _clean_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
        """Remove compile-related prefixes from state dict keys."""
        prefix = "_orig_mod."
        return {key[len(prefix) :] if key.startswith(prefix) else key: value for key, value in state_dict.items()}

    @staticmethod
    def _get_underlying_model(model: torch.nn.Module) -> torch.nn.Module:
        """Get underlying model, unwrapping compiled models."""
        return getattr(model, "_orig_mod", model)

    @classmethod
    def from_model(cls, model: torch.nn.Module) -> HyformerCheckpoint:
        """Create checkpoint from a model, removing compile artifacts.

        Parameters
        ----------
        model : torch.nn.Module
            Model to create checkpoint from. May be compiled.

        Returns
        -------
        HyformerCheckpoint
            Checkpoint instance with cleaned state dict.
        """
        model = cls._get_underlying_model(model)
        state_dict = cls._clean_state_dict_keys(model.state_dict())
        return cls(state_dict=state_dict)

    def save(
        self,
        path: str | Path,
        safe_serialization: bool = True,
    ) -> None:
        """Save checkpoint to file.

        Follows HuggingFace best practices:
        - Uses safetensors format by default (safer, faster, cross-platform)
        - Falls back to PyTorch format if safetensors unavailable

        Parameters
        ----------
        path : str | Path
            Path to save checkpoint file. If directory, saves as model.safetensors
            or pytorch_model.bin. If file, uses the specified extension.
        safe_serialization : bool, default=True
            If True, saves in safetensors format (recommended). If False or
            safetensors unavailable, saves in PyTorch format.
        """
        path = Path(path)

        # If directory, use HF standard filename
        if path.is_dir() or not path.suffix:
            path.mkdir(parents=True, exist_ok=True)
            if safe_serialization and _SAFETENSORS_AVAILABLE:
                save_path = path / "model.safetensors"
            else:
                save_path = path / "pytorch_model.bin"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            save_path = path

        # Save in safetensors format if requested and available
        if safe_serialization and _SAFETENSORS_AVAILABLE and save_path.suffix == ".safetensors":
            safetensors_save(self.state_dict, save_path)
        elif save_path.suffix == ".safetensors" and not _SAFETENSORS_AVAILABLE:
            # User requested safetensors but library not available
            raise ImportError("Saving safetensors requires safetensors library. Install with: pip install safetensors")
        else:
            # Save in PyTorch format
            torch.save(self.state_dict, save_path)

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str | None = None) -> HyformerCheckpoint:
        """Load checkpoint from local file.

        Supports both PyTorch (.pt, .pth, .bin) and safetensors formats.
        """
        path = Path(path)

        # If directory, look for standard HF checkpoint files
        if path.is_dir():
            # Try safetensors first (HF preferred format)
            safetensors_path = path / "model.safetensors"
            if safetensors_path.exists():
                if not _SAFETENSORS_AVAILABLE:
                    raise ImportError(
                        "Loading safetensors requires safetensors library. Install with: pip install safetensors"
                    )
                return cls(safetensors_load(safetensors_path))

            # Fall back to PyTorch format
            pytorch_path = path / "pytorch_model.bin"
            if pytorch_path.exists():
                state_dict = torch.load(pytorch_path, map_location=device)
                return cls(state_dict=state_dict)

            raise FileNotFoundError(f"No checkpoint found in {path}. Expected model.safetensors or pytorch_model.bin")

        # Load from file
        if path.suffix == ".safetensors":
            if not _SAFETENSORS_AVAILABLE:
                raise ImportError(
                    "Loading safetensors requires safetensors library. Install with: pip install safetensors"
                )
            state_dict = safetensors_load(path)
        else:
            state_dict = torch.load(path, map_location=device)
        return cls(state_dict=state_dict)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        filename: str | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        device: torch.device | str | None = None,
    ) -> HyformerCheckpoint:
        """Load checkpoint from HuggingFace Hub.

        Parameters
        ----------
        model_id : str
            HuggingFace model identifier (e.g., "username/model-name").
        filename : str | None, default=None
            Specific filename to load. If None, tries: pytorch_model.bin,
            model.safetensors, model.pt, model.pth.
        revision : str | None, default=None
            Git revision (branch, tag, or commit hash).
        cache_dir : str | None, default=None
            Directory to cache downloaded files.
        device : torch.device | str | None, default=None
            Device to load checkpoint on. If None, loads on CPU.

        Returns
        -------
        HyformerCheckpoint
            Checkpoint instance with loaded state dict.

        Raises
        ------
        ImportError
            If huggingface_hub is not installed.
        FileNotFoundError
            If no checkpoint file is found.
        """
        if not _HF_AVAILABLE:
            raise ImportError(
                "Loading from HuggingFace requires huggingface_hub. Install with: pip install huggingface_hub"
            )

        if filename:
            file_path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                revision=revision,
                cache_dir=cache_dir,
            )
            return cls.load(file_path, device=device)

        # Try common checkpoint filenames (prefer safetensors - HF best practice)
        for filename in ["model.safetensors", "pytorch_model.bin", "model.pt", "model.pth"]:
            try:
                file_path = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    revision=revision,
                    cache_dir=cache_dir,
                )
                logger.info("Loaded checkpoint from HuggingFace: {} ({})", model_id, filename)
                return cls.load(file_path, device=device)
            except Exception:
                continue

        raise FileNotFoundError(
            f"No checkpoint file found in {model_id}. Tried: pytorch_model.bin, model.safetensors, model.pt, model.pth"
        )
