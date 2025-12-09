"""Hit Ratio metric for evaluating molecular optimization performance.

The Hit Ratio measures the proportion of generated molecules that meet a
specified threshold criterion. This metric is commonly used in molecular
optimization to assess the effectiveness of generative models in producing
molecules with desired properties.

This implementation follows SATURN's Hit Ratio calculation methodology.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


def calculate_hit_ratio(
    scores: np.ndarray | Sequence[float],
    threshold: float,
    minimize: bool = True,
) -> float:
    """Calculate the Hit Ratio for a set of molecular scores.

    The Hit Ratio is defined as the proportion of generated molecules that
    meet or exceed (for maximization) or meet or fall below (for minimization)
    a specified threshold value.

    Parameters
    ----------
    scores : np.ndarray | Sequence[float]
        Array or sequence of property scores for generated molecules.
        Shape: (n_molecules,). Can contain NaN values which are excluded.
    threshold : float
        Threshold value that defines a "hit". For minimization (default),
        molecules with scores <= threshold are hits. For maximization,
        molecules with scores >= threshold are hits.
    minimize : bool, default=True
        If True, treats lower scores as better (e.g., docking scores).
        If False, treats higher scores as better (e.g., drug-likeness scores).

    Returns
    -------
    float
        Hit Ratio value between 0.0 and 1.0. Represents the fraction of
        valid molecules that meet the threshold criterion.

    Examples
    --------
    >>> import numpy as np
    >>> # Docking scores (lower is better, threshold = -8.0)
    >>> docking_scores = np.array([-8.5, -7.2, -9.1, -8.0, -6.5])
    >>> hit_ratio = calculate_hit_ratio(docking_scores, threshold=-8.0, minimize=True)
    >>> hit_ratio  # Molecules with score <= -8.0: [-8.5, -9.1, -8.0] = 3/5 = 0.6
    0.6

    >>> # Drug-likeness scores (higher is better, threshold = 0.5)
    >>> drug_scores = np.array([0.6, 0.4, 0.7, 0.3, 0.5])
    >>> hit_ratio = calculate_hit_ratio(drug_scores, threshold=0.5, minimize=False)
    >>> hit_ratio  # Molecules with score >= 0.5: [0.6, 0.7, 0.5] = 3/5 = 0.6
    0.6

    >>> # Handle NaN values
    >>> scores_with_nan = np.array([-8.5, np.nan, -9.1, -8.0, np.nan])
    >>> hit_ratio = calculate_hit_ratio(scores_with_nan, threshold=-8.0, minimize=True)
    >>> hit_ratio  # Valid scores: [-8.5, -9.1, -8.0], hits: [-8.5, -9.1, -8.0] = 3/3 = 1.0
    1.0

    Notes
    -----
    - NaN values in scores are automatically excluded from the calculation.
    - If all scores are NaN, returns 0.0.
    - The threshold comparison is inclusive (scores equal to threshold count as hits).
    - This implementation follows SATURN's Hit Ratio methodology for molecular
      optimization evaluation.

    References
    ----------
    - SATURN framework: https://github.com/schwallergroup/saturn
    - Common usage: Hit Ratio@threshold measures fraction of molecules meeting
      property threshold (e.g., docking score <= -8.0 kcal/mol)
    """
    scores = np.asarray(scores, dtype=np.float64)

    if scores.ndim != 1:  # type: ignore[union-attr]
        raise ValueError(f"scores must be 1D array or sequence, got shape {scores.shape}")  # type: ignore[union-attr]

    if not np.isfinite(threshold):
        raise ValueError(f"threshold must be a finite number, got {threshold}")

    # Filter out NaN and infinite values
    valid_mask = np.isfinite(scores)
    valid_scores = scores[valid_mask]

    if len(valid_scores) == 0:
        # No valid scores, return 0.0
        return 0.0

    # Determine hits based on minimization/maximization mode
    if minimize:
        # For minimization: hits are scores <= threshold (lower is better)
        hits: int = int(np.sum(valid_scores <= threshold))
    else:
        # For maximization: hits are scores >= threshold (higher is better)
        hits = np.sum(valid_scores >= threshold)

    hit_ratio = hits / len(valid_scores)

    return float(hit_ratio)
