"""Configuration for Hyformer backbone model."""

from __future__ import annotations

from dataclasses import dataclass

from joint_improvement.utils.config import BaseConfig


@dataclass
class HyformerConfig(BaseConfig):
    """Configuration for Hyformer transformer model.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size (number of tokens).
    d_model : int, default=512
        Model dimension (hidden size). Must be divisible by n_heads.
    n_heads : int, default=8
        Number of attention heads.
    n_layers : int, default=6
        Number of transformer layers.
    max_seq_len : int, default=128
        Maximum sequence length.
    attn_dropout : float, default=0.0
        Attention dropout probability.
    resid_dropout : float, default=0.0
        Residual dropout probability.
    eps : float, default=1e-6
        RMSNorm epsilon value.
    num_prediction_tasks : int | None, default=None
        Number of prediction task outputs. If None, prediction head is disabled.
    prediction_task_type : str | None, default=None
        Explicit type of the "prediction" task to select the correct loss.
        If None, the loss is inferred from targets dtype/shape.
        Supported values:
        - "multitarget_regression"
        - "regression"
        - "classification"
        - "multilabel_classification"
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
    prediction_task_type: str | None = None
    generator_type: str = "unconditional"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
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
        if self.prediction_task_type is not None:
            allowed = {
                "multitarget_regression",
                "regression",
                "classification",
                "multilabel_classification",
            }
            if self.prediction_task_type not in allowed:
                raise ValueError(
                    f"prediction_task_type must be one of {sorted(allowed)}, got {self.prediction_task_type!r}"
                )
        if self.generator_type not in ("unconditional", "tasar", "tasar_legacy"):
            raise ValueError(
                f"generator_type must be one of 'unconditional', 'tasar', 'tasar_legacy', got {self.generator_type}"
            )
