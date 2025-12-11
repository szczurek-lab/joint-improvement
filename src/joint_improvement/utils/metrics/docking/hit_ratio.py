"""Hit Ratio metric for evaluating molecular optimization performance.

The Hit Ratio measures the proportion of generated molecules that meet a
specified threshold criterion. This metric is commonly used in molecular
optimization to assess the effectiveness of generative models in producing
molecules with desired properties.

This implementation follows SATURN's Hit Ratio calculation methodology.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence


def calculate_hit_ratio(
    docking_scores: NDArray[np.floating[Any]] | Sequence[float],
    qed_scores: NDArray[np.floating[Any]] | Sequence[float],
    sa_scores: NDArray[np.floating[Any]] | Sequence[float],
    docking_threshold: float,
    qed_threshold: float = 0.5,
    sa_threshold: float = 5.0,
) -> float:
    """Calculate hit ratio requiring all three property conditions simultaneously.

    Conditions (inclusive):
    - Docking score > ``docking_threshold`` (higher is better for this check)
    - QED score >= ``qed_threshold`` (higher is better)
    - SA score <= ``sa_threshold`` (lower is better)

    Parameters
    ----------
    docking_scores : np.ndarray | Sequence[float]
        Docking scores (1D).
    qed_scores : np.ndarray | Sequence[float]
        QED scores (1D).
    sa_scores : np.ndarray | Sequence[float]
        SA scores (1D).
    docking_threshold : float
        Threshold for docking scores (strictly greater than).
    qed_threshold : float
        Threshold for QED scores (greater or equal).
    sa_threshold : float
        Threshold for SA scores (less or equal).

    Returns
    -------
    float
        Fraction of molecules satisfying all three conditions. Returns 0.0 if
        there are no valid (finite) triplets.

    Notes
    -----
    NaN/inf entries are filtered out per-position; only positions where all three
    scores are finite are considered.
    """
    docking_scores_arr: NDArray[np.float64] = np.asarray(docking_scores, dtype=np.float64)
    qed_scores_arr: NDArray[np.float64] = np.asarray(qed_scores, dtype=np.float64)
    sa_scores_arr: NDArray[np.float64] = np.asarray(sa_scores, dtype=np.float64)

    if docking_scores_arr.ndim != 1 or qed_scores_arr.ndim != 1 or sa_scores_arr.ndim != 1:
        raise ValueError("All score arrays must be 1D.")

    if not (docking_scores_arr.shape[0] == qed_scores_arr.shape[0] == sa_scores_arr.shape[0]):
        raise ValueError("All score arrays must have the same length.")

    valid_mask = np.isfinite(docking_scores_arr) & np.isfinite(qed_scores_arr) & np.isfinite(sa_scores_arr)

    if not np.any(valid_mask):
        return 0.0

    d_valid = docking_scores_arr[valid_mask]
    q_valid = qed_scores_arr[valid_mask]
    s_valid = sa_scores_arr[valid_mask]

    hits = (d_valid < docking_threshold) & (q_valid > qed_threshold) & (s_valid < sa_threshold)

    return float(np.sum(hits) / len(hits))
