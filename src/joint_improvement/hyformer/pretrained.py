"""MixIn for pretrained model loading and saving functionality."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

import torch
from loguru import logger  # type: ignore

from .checkpoint import HyformerCheckpoint

if TYPE_CHECKING:
    from .config import HyformerConfig

try:
    from huggingface_hub import hf_hub_download

    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    hf_hub_download = None

try:
    from safetensors.torch import save_file as safetensors_save

    _SAFETENSORS_AVAILABLE = True
except ImportError:
    _SAFETENSORS_AVAILABLE = False
    safetensors_save = None


class PretrainedMixin:
    """MixIn providing from_pretrained and save_pretrained methods.

    This MixIn can be used with any model class that has:
    - A config class with `to_dict()` method
    - A constructor that accepts config parameters
    - A `from_config` classmethod
    """

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        config: HyformerConfig | None = None,
        filename: str | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        device: torch.device | str | None = None,
        strict: bool = True,
    ):
        """Load model from HuggingFace Hub checkpoint.

        Parameters
        ----------
        model_id : str
            HuggingFace model identifier (e.g., "username/model-name").
        config : HyformerConfig | None, default=None
            Model configuration. If None, must be provided in the checkpoint.
        filename : str | None, default=None
            Specific checkpoint filename to load.
        revision : str | None, default=None
            Git revision (branch, tag, or commit hash).
        cache_dir : str | None, default=None
            Directory to cache downloaded files.
        device : torch.device | str | None, default=None
            Device to load checkpoint on. If None, loads on CPU.
        strict : bool, default=True
            Whether to strictly enforce that the checkpoint keys match model keys.

        Returns
        -------
        Model instance
            Loaded model.

        Examples
        --------
        >>> model = Hyformer.from_pretrained("username/hyformer-model")
        >>> model = Hyformer.from_pretrained(
        ...     "username/hyformer-model", config=HyformerConfig(vocab_size=32000, d_model=512), strict=False
        ... )
        """
        # Load checkpoint from HF
        checkpoint = HyformerCheckpoint.from_pretrained(
            model_id=model_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            device=device,
        )

        # Load config if not provided
        if config is None:
            # Try to load config.json from HF Hub
            if _HF_AVAILABLE:
                try:
                    import json

                    from .config import HyformerConfig

                    config_path = hf_hub_download(
                        repo_id=model_id,
                        filename="config.json",
                        revision=revision,
                        cache_dir=cache_dir,
                    )
                    with open(config_path) as f:
                        config_dict = json.load(f)
                    config = HyformerConfig(**config_dict)
                    logger.info("Loaded config from HuggingFace: {}", model_id)
                except Exception:
                    pass

            if config is None:
                raise ValueError(
                    "config must be provided. Could not load config.json from checkpoint. "
                    "Provide HyformerConfig when loading from checkpoint."
                )

        # Create model from config
        # Prefer from_config if available, otherwise use config.to_dict()
        if hasattr(cls, "from_config"):
            model = cls.from_config(config)
        else:
            # Fallback: try to instantiate directly with config dict
            # Only pass parameters that the model's __init__ accepts
            config_dict = config.to_dict()
            # Filter to only include vocab_size, d_model, n_heads, n_layers, max_seq_len
            model_params = {
                "vocab_size": config_dict["vocab_size"],
                "d_model": config_dict["d_model"],
                "n_heads": config_dict["n_heads"],
                "n_layers": config_dict["n_layers"],
                "max_seq_len": config_dict["max_seq_len"],
            }
            model = cls(**model_params)

        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint.state_dict,
            strict=strict,
        )

        if not strict:
            if missing_keys:
                logger.warning("Missing keys when loading checkpoint: {}", missing_keys)
            if unexpected_keys:
                logger.warning("Unexpected keys in checkpoint: {}", unexpected_keys)

        if device:
            model = model.to(device)

        return model

    def save_pretrained(
        self,
        save_directory: str | Path,
        safe_serialization: bool = True,
    ) -> None:
        """Save model following HuggingFace best practices.

        Saves model weights and config in a directory structure compatible
        with HuggingFace Hub. Uses safetensors format by default (recommended).

        Parameters
        ----------
        save_directory : str | Path
            Directory to save model files. Will create model.safetensors (or
            pytorch_model.bin) and config.json.
        safe_serialization : bool, default=True
            If True, saves weights in safetensors format (recommended).
            If False, saves in PyTorch format.

        Examples
        --------
        >>> model.save_pretrained("./my_model")
        >>> # Creates: ./my_model/model.safetensors and ./my_model/config.json
        """
        import json
        from pathlib import Path

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        checkpoint = HyformerCheckpoint.from_model(self)
        checkpoint.save(save_dir, safe_serialization=safe_serialization)

        # Extract config from model structure
        # Note: Some parameters (attn_dropout, etc.) use defaults
        # as they're not stored in the model after initialization
        first_block = self.blocks[0]  # type: ignore[attr-defined]
        config_dict = {
            "vocab_size": self.embed.num_embeddings,  # type: ignore[attr-defined]
            "d_model": self.embed.embedding_dim,  # type: ignore[attr-defined]
            "n_heads": first_block.attn.n_heads,  # type: ignore[attr-defined]
            "n_layers": len(self.blocks),  # type: ignore[attr-defined]
            "max_seq_len": first_block.attn.max_seq_len,  # type: ignore[attr-defined]
            "attn_dropout": 0.0,
            "resid_dropout": 0.0,
            "eps": getattr(first_block.input_norm, "eps", 1e-6),
            "num_prediction_tasks": getattr(self, "num_prediction_tasks", None),
        }

        # Try to extract dropout values if stored
        if hasattr(first_block.attn, "attn_dropout_p"):
            config_dict["attn_dropout"] = first_block.attn.attn_dropout_p
        if hasattr(first_block, "resid_dropout"):
            if hasattr(first_block.resid_dropout, "p"):
                config_dict["resid_dropout"] = first_block.resid_dropout.p

        # Save config as JSON (HF standard)
        config_path = save_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(
            "Saved model to {} (format: {})",
            save_dir,
            "safetensors" if safe_serialization and _SAFETENSORS_AVAILABLE else "pytorch",
        )
