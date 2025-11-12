"""Target transformation utilities for sequence datasets.

This module provides target transformation functions and classes for
normalizing and scaling target values in chemistry ML datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


class StandardScaler:
    """Standard scaler (z-score normalization) for target values.

    Transforms targets to have zero mean and unit variance. Useful for regression
    tasks where targets have different scales or distributions.

    Examples:
    --------
    >>> import numpy as np
    >>> targets = [10.0, 20.0, 30.0, 40.0, 50.0]
    >>> scaler = StandardScaler.fit(targets)
    >>> scaler(25.0)
    0.0
    >>> scaler.inverse_transform(0.0)
    25.0
    """

    def __init__(self, mean: float, std: float) -> None:
        """Initialize StandardScaler with precomputed statistics.

        Parameters
        ----------
        mean : float
            Mean of the target values.
        std : float
            Standard deviation of the target values.
        """
        self.mean = mean
        self.std = std if std > 0 else 1.0  # Avoid division by zero

    def __call__(self, value: float | int) -> float:
        """Transform a single target value.

        Parameters
        ----------
        value : float | int
            Target value to transform.

        Returns:
        -------
        float
            Standardized target value.
        """
        return float((value - self.mean) / self.std)

    def inverse_transform(self, value: float) -> float:
        """Inverse transform a standardized value back to original scale.

        Parameters
        ----------
        value : float
            Standardized value to transform back.

        Returns:
        -------
        float
            Value in original scale.
        """
        return float(value * self.std + self.mean)

    def to_dict(self) -> dict[str, Any]:
        """Serialize scaler to dictionary.

        Returns:
        -------
        dict[str, Any]
            Dictionary containing transform type and parameters.
        """
        return {
            "type": "standard",
            "mean": self.mean,
            "std": self.std,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> StandardScaler:
        """Create scaler from configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary with "mean" and "std" keys.

        Returns:
        -------
        StandardScaler
            Scaler instance created from config.

        Examples:
        --------
        >>> config = {"type": "standard", "mean": 25.0, "std": 10.0}
        >>> scaler = StandardScaler.from_dict(config)
        >>> scaler(25.0)
        0.0
        """
        return cls(mean=config["mean"], std=config["std"])

    @classmethod
    def fit(cls, targets: list[float | int] | np.ndarray) -> StandardScaler:
        """Fit scaler on target values.

        Parameters
        ----------
        targets : list[float | int] | np.ndarray
            Target values to compute statistics from.

        Returns:
        -------
        StandardScaler
            Fitted scaler instance.
        """
        arr = np.asarray(targets, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        return cls(mean=mean, std=std)


def create_target_transform(
    transform_type: str | None = None,
    targets: list[float | int] | np.ndarray | None = None,
    config: dict[str, Any] | None = None,
) -> Callable[[float | int], float]:
    """Factory function to create target transforms.

    Currently only supports StandardScaler. Can be called in multiple ways:
    1. With transform_type string and targets
    2. With a config dictionary (from JSON/config file)

    Parameters
    ----------
    transform_type : str | None, optional
        Type of transform to create. Currently only "standard" is supported.
        Ignored if config is provided.
    targets : list[float | int] | np.ndarray | None, optional
        Target values for fitting scaler. Required when using transform_type.
    config : dict[str, Any] | None, optional
        Configuration dictionary. If provided, transform_type is ignored.
        Dictionary should have "type": "standard", "mean", and "std" keys.

    Returns:
    -------
    Callable[[float | int], float]
        Target transform function.

    Raises:
    ------
    ValueError
        If transform_type is not recognized or targets are required but not provided.

    Examples:
    --------
    Using string type with targets:
    >>> targets = [10.0, 20.0, 30.0]
    >>> transform = create_target_transform("standard", targets=targets)
    >>> transform(20.0)
    0.0

    Using config dictionary (from JSON):
    >>> config = {"type": "standard", "mean": 20.0, "std": 10.0}
    >>> transform = create_target_transform(config=config)
    >>> transform(20.0)
    0.0
    """
    # Handle config dictionary (from JSON/config file)
    if config is not None:
        transform_config = config
        transform_type_str = transform_config.get("type")
        if transform_type_str == "standard":
            return StandardScaler.from_dict(transform_config)
        else:
            raise ValueError(f"Unknown transform type '{transform_type_str}' in config. Supported type: 'standard'")

    # Handle string-based creation
    if transform_type is None:
        raise ValueError("Either 'transform_type' or 'config' must be provided")

    if transform_type == "standard":
        if targets is None:
            raise ValueError("targets must be provided for 'standard' transform")
        return StandardScaler.fit(targets)
    else:
        raise ValueError(f"Unknown transform_type '{transform_type}'. Supported type: 'standard'")
