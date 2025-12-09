"""Trainer checkpoint dataclass and mixin."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .trainer import TrainerState


@dataclass
class TrainerCheckpoint:
    """Trainer checkpoint containing optimizer state and training metadata."""

    model_state_dict: dict[str, Any]
    optimizer_state: dict[str, Any]
    trainer_state: TrainerState


class TrainerCheckpointMixin:
    """Mixin providing trainer checkpoint save/load functionality.

    Requires: model, optimizer, state (TrainerState), device, out_dir attributes.
    """

    def _get_underlying_model(self) -> torch.nn.Module:
        """Get underlying model, unwrapping compiled models to avoid compile artifacts.

        Returns
        -------
        torch.nn.Module
            The underlying model without compilation wrapper.
        """
        if hasattr(self.model, "_orig_mod"):  # type: ignore[attr-defined]
            # Model is compiled with torch.compile(), use original model
            return self.model._orig_mod  # type: ignore[attr-defined]
        return self.model  # type: ignore[attr-defined]

    @staticmethod
    def _clean_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
        """Remove compile-related prefixes from state dict keys.

        Parameters
        ----------
        state_dict : dict[str, Any]
            State dict that may contain keys with '_orig_mod.' prefix.

        Returns
        -------
        dict[str, Any]
            State dict with cleaned keys (no '_orig_mod.' prefix).
        """
        cleaned = {}
        prefix = "_orig_mod."
        for key, value in state_dict.items():
            if key.startswith(prefix):
                cleaned[key[len(prefix) :]] = value
            else:
                cleaned[key] = value
        return cleaned

    def save_trainer_checkpoint(self, path: str | Path) -> None:
        """Save trainer checkpoint."""
        # Get underlying model to avoid saving compile artifacts
        model_to_save = self._get_underlying_model()
        # Get state dict and clean keys to remove any compile-related prefixes
        model_state_dict = model_to_save.state_dict()
        model_state_dict = self._clean_state_dict_keys(model_state_dict)

        checkpoint = TrainerCheckpoint(
            model_state_dict=model_state_dict,
            optimizer_state=self.optimizer.state_dict(),  # type: ignore[attr-defined]
            trainer_state=self.state,  # type: ignore[attr-defined]
        )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": checkpoint.model_state_dict,
                "optimizer": checkpoint.optimizer_state,
                "epoch": checkpoint.trainer_state.epoch,
                "best_val_loss": checkpoint.trainer_state.best_val_loss,
            },
            path,
        )

    def load_trainer_checkpoint(self, path: str | Path) -> None:
        """Load trainer checkpoint."""
        from .trainer import TrainerState

        data = torch.load(path, map_location=getattr(self, "device", None))
        checkpoint = TrainerCheckpoint(
            model_state_dict=data.get("model", {}),
            optimizer_state=data["optimizer"],
            trainer_state=TrainerState(
                epoch=data.get("epoch", 0),
                best_val_loss=data.get("best_val_loss", float("inf")),
            ),
        )
        if checkpoint.model_state_dict:
            # Load into underlying model to ensure correct loading
            model_to_load = self._get_underlying_model()
            # Clean keys in case checkpoint was saved with compile prefixes
            cleaned_state_dict = self._clean_state_dict_keys(checkpoint.model_state_dict)
            model_to_load.load_state_dict(cleaned_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state)  # type: ignore[attr-defined]
        self.state = checkpoint.trainer_state  # type: ignore[attr-defined]

    def _save_checkpoint(self, is_best: bool = False, epoch: int | None = None) -> None:
        """Save full checkpoint (model + trainer).

        Saves best checkpoint as "best.pt" (always overwrites).
        Optionally saves periodic checkpoint if epoch is provided.

        Parameters
        ----------
        is_best : bool, optional
            Whether this is the best checkpoint so far. Saves as "best.pt".
        epoch : int, optional
            Epoch number for periodic checkpoint. If provided, saves as "checkpoint_epoch_{epoch}.pt".
        """
        if not self.out_dir:  # type: ignore[attr-defined]
            return

        if is_best:
            # Always save as "best.pt" (single checkpoint, overwrites on update)
            filename = "best.pt"
            path = self.out_dir / filename  # type: ignore[attr-defined]
        elif epoch is not None:
            # Save periodic checkpoint
            filename = f"checkpoint_epoch_{epoch}.pt"
            path = self.out_dir / filename  # type: ignore[attr-defined]
        else:
            # No checkpoint to save
            return

        # Use mixin to get trainer checkpoint data format
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pt", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            # Delegate trainer checkpoint saving to mixin to get consistent format
            self.save_trainer_checkpoint(tmp_path)
            trainer_data = torch.load(tmp_path, map_location=self.device)  # type: ignore[attr-defined]

            # Save full checkpoint (model is already in trainer_data from save_trainer_checkpoint)
            torch.save(trainer_data, path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _load_checkpoint(self, path: str | Path, resume_training: bool = True) -> None:
        """Load full checkpoint (model + trainer).

        Parameters
        ----------
        path : str | Path
            Path to checkpoint file.
        resume_training : bool, optional
            Whether to resume training state.
        """
        data = torch.load(path, map_location=self.device)  # type: ignore[attr-defined]

        # Load model if present
        if "model" in data:
            # Load into underlying model to ensure correct loading
            model_to_load = self._get_underlying_model()
            # Clean keys in case checkpoint was saved with compile prefixes
            cleaned_state_dict = self._clean_state_dict_keys(data["model"])
            model_to_load.load_state_dict(cleaned_state_dict)

        if resume_training:
            # Delegate trainer checkpoint loading to mixin
            # Create temporary checkpoint file with trainer data (excluding model, already loaded)
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".pt", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)

            try:
                torch.save(
                    {
                        "model": {},  # Model already loaded, pass empty dict
                        "optimizer": data["optimizer"],
                        "epoch": data.get("epoch", 0),
                        "best_val_loss": data.get("best_val_loss", float("inf")),
                    },
                    tmp_path,
                )
                self.load_trainer_checkpoint(tmp_path)
            finally:
                tmp_path.unlink(missing_ok=True)
