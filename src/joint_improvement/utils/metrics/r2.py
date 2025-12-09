"""R² indicator for multi-objective optimization evaluation.

The R² indicator evaluates the quality of a set of solutions in multi-objective
optimization by measuring the average distance from reference points to the
solutions. Lower values indicate better solution sets.

This implementation is based on MolStitch evaluators:
https://github.com/MolecularTeam/MolStitch/blob/main/evaluators/utils.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


def calculate_r2(
    reference_points: np.ndarray | Sequence[Sequence[float]],
    solutions: np.ndarray | Sequence[Sequence[float]],
    utopian_point: np.ndarray | Sequence[float],
) -> float:
    """Calculate the R² indicator value for a set of solutions.

    The R² indicator computes the average minimum distance from reference points
    to the solution set, normalized by the norm of the reference points. This
    metric is used to evaluate the quality of Pareto-optimal solutions in
    multi-objective optimization.

    Parameters
    ----------
    reference_points : np.ndarray | Sequence[Sequence[float]]
        Array of reference points from a uniform distribution, shape
        (n_reference_points, n_objectives). These represent different preference
        vectors for evaluating solution quality.
    solutions : np.ndarray | Sequence[Sequence[float]]
        Multi-objective solutions (fitness values), shape (n_solutions, n_objectives).
        Typically these are Pareto-optimal solutions.
    utopian_point : np.ndarray | Sequence[float]
        Utopian point representing the best possible solution, shape (n_objectives,).
        This is the ideal point that represents the best value for each objective.

    Returns
    -------
    float
        The R² indicator value. Lower values indicate better solution sets.

    Examples
    --------
    >>> import numpy as np
    >>> # 2-objective problem
    >>> reference_points = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    >>> solutions = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]])
    >>> utopian_point = np.array([0.0, 0.0])
    >>> r2 = calculate_r2(reference_points, solutions, utopian_point)
    >>> r2 >= 0
    True

    Notes
    -----
    - The R² indicator is computed as the average minimum weighted Chebyshev
      distance from reference points to solutions.
    - Reference points are typically sampled uniformly from a unit simplex or
      hyperplane to represent different preference vectors.
    - The utopian point should represent the best possible value for each
      objective (typically the minimum for minimization problems).
    - Lower R² values indicate better solution sets that are closer to the
      reference points.
    """
    reference_points = np.asarray(reference_points, dtype=np.float64)
    solutions = np.asarray(solutions, dtype=np.float64)
    utopian_point = np.asarray(utopian_point, dtype=np.float64)

    if reference_points.ndim != 2:  # type: ignore[union-attr]
        raise ValueError(f"reference_points must be 2D array, got shape {reference_points.shape}")  # type: ignore[union-attr]
    if solutions.ndim != 2:  # type: ignore[union-attr]
        raise ValueError(f"solutions must be 2D array, got shape {solutions.shape}")  # type: ignore[union-attr]
    if utopian_point.ndim != 1:  # type: ignore[union-attr]
        raise ValueError(f"utopian_point must be 1D array, got shape {utopian_point.shape}")  # type: ignore[union-attr]

    n_reference, n_obj_ref = reference_points.shape  # type: ignore[union-attr]
    n_solutions, n_obj_sol = solutions.shape  # type: ignore[union-attr]

    if n_obj_ref != n_obj_sol:
        raise ValueError(
            f"reference_points and solutions must have same number of objectives: {n_obj_ref} != {n_obj_sol}"
        )
    if utopian_point.shape[0] != n_obj_ref:  # type: ignore[union-attr]
        raise ValueError(
            f"utopian_point must have same number of objectives as reference_points: "
            f"{utopian_point.shape[0]} != {n_obj_ref}"  # type: ignore[union-attr]
        )

    if n_reference == 0:
        raise ValueError("reference_points must contain at least one point")
    if n_solutions == 0:
        raise ValueError("solutions must contain at least one solution")

    # Compute R² indicator following MolStitch implementation
    min_list: list[float] = []
    for v in reference_points:
        max_list: list[float] = []
        for a in solutions:
            # Compute weighted Chebyshev distance: max(v * |utopian_point - a|)
            weighted_distance: float = float(np.max(v * np.abs(utopian_point - a)))  # type: ignore[operator]
            max_list.append(weighted_distance)
        # Find minimum distance for this reference point
        min_list.append(np.min(max_list))

    # Normalize by the norm of reference points
    v_norm = np.linalg.norm(reference_points)
    if v_norm == 0:
        raise ValueError("reference_points norm is zero, cannot compute R²")

    r2 = np.sum(min_list) / v_norm

    return float(r2)
