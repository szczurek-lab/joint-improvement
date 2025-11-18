"""Target transformation utilities for sequence datasets.

This module provides target transformation functions and classes for
normalizing and scaling target values in chemistry ML datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class ZScoreScaler:
    """Z-score normalization (standardization) scaler for target values.

    Transforms targets using z-score normalization: (x - mean) / std.
    """

    def __init__(self, mean: float, std: float) -> None:
        """Initialize ZScoreScaler with precomputed statistics.

        Parameters
        ----------
        mean : float
            Mean of the target values.
        std : float
            Standard deviation of the target values.
        """
        self.mean = mean
        self.std = std

    def __call__(self, value: float | int) -> float:
        """Transform a single target value using z-score normalization.

        Parameters
        ----------
        value : float | int
            Target value to transform.

        Returns
        -------
        float
            Z-score normalized target value: (value - mean) / std.
        """
        return float((value - self.mean) / self.std)

    def inverse_transform(self, value: float) -> float:
        """Inverse transform a z-score normalized value back to original scale.

        Parameters
        ----------
        value : float
            Z-score normalized value to transform back.

        Returns
        -------
        float
            Value in original scale: value * std + mean.
        """
        return float(value * self.std + self.mean)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ZScoreScaler:
        """Create scaler from configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary with "type": "zscore", "mean", and "std" keys.

        Returns
        -------
        ZScoreScaler
            Scaler instance created from config.
        """
        return cls(mean=config["mean"], std=config["std"])


def create_target_transform(config: dict[str, Any]) -> Callable[[float | int], float]:
    """Create target transform from configuration dictionary.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary with "type": "zscore", "mean", and "std" keys.

    Returns
    -------
    Callable[[float | int], float]
        Target transform function.

    Raises
    ------
    ValueError
        If transform type is not recognized or mean/std are missing.
    """
    transform_type = config.get("type")
    if transform_type not in ("zscore"):
        raise ValueError(f"Unknown transform type '{transform_type}'. Supported types: 'zscore'")

    if "mean" not in config or "std" not in config:
        raise ValueError(f"For '{transform_type}' transform, 'mean' and 'std' must be provided in config")

    return ZScoreScaler.from_config(config)
