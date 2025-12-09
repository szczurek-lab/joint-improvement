"""Configuration for Hyformer backbone model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from joint_improvement.utils.config import BaseConfig


@dataclass
class HyformerConfig(BaseConfig):
    """
    Configuration for Hyformer transformer model.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary (number of tokens).
    d_model : int, default=512
        Model dimension (hidden size). Should be divisible by n_heads.
    n_heads : int, default=8
        Number of attention heads. d_model must be divisible by n_heads.
    n_layers : int, default=6
        Number of decoder blocks (transformer layers).
    max_seq_len : int, default=128
        Maximum sequence length for positional encodings and attention masks.
    attn_dropout : float, default=0.0
        Dropout probability for attention weights.
    resid_dropout : float, default=0.0
        Dropout probability for residual connections.
    eps : float, default=1e-6
        Epsilon value for RMSNorm layers.
    num_prediction_tasks : int | None, default=None
        Number of outputs for prediction head. If None, prediction head is disabled.

    Attributes
    ----------
    vocab_size : int
        Vocabulary size.
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of decoder layers.
    max_seq_len : int
        Maximum sequence length.
    attn_dropout : float
        Attention dropout probability.
    resid_dropout : float
        Residual dropout probability.
    eps : float
        RMSNorm epsilon value.

    """

    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    max_seq_len: int = 128
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    eps: float = 1e-6
    num_prediction_tasks: int | None = None

    def __post_init__(self) -> None:
        """
        Validate configuration parameters.

        Raises
        ------
        ValueError
            If d_model is not divisible by n_heads, or if any dimension
            is non-positive.
        """
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.num_prediction_tasks is not None and self.num_prediction_tasks <= 0:
            raise ValueError(f"num_prediction_tasks must be positive when set, got {self.num_prediction_tasks}")

    def to_dict(self) -> dict[str, int | float]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        dict[str, int | float]
            Dictionary containing all configuration parameters.
        """
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "max_seq_len": self.max_seq_len,
            "attn_dropout": self.attn_dropout,
            "resid_dropout": self.resid_dropout,
            "eps": self.eps,
            "num_prediction_tasks": self.num_prediction_tasks,
        }

    @classmethod
    def from_json(cls, path: str | Path) -> HyformerConfig:
        """Load configuration from JSON file.

        Parameters
        ----------
        path : str | Path
            Path to JSON configuration file.

        Returns
        -------
        HyformerConfig
            Configuration instance loaded from JSON.

        Examples
        --------
        >>> config = HyformerConfig.from_json("configs/hyformer/base.json")
        """
        import json

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls(**config_dict)
