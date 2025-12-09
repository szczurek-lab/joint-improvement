"""Hypervolume (HV) metric for multi-objective optimization evaluation.

The hypervolume metric measures the volume of the objective space dominated by
a set of solutions relative to a reference point. It is widely used in
multi-objective optimization to assess the quality of solution sets.

This implementation is inspired by MolStitch evaluators.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


def calculate_hypervolume(
    points: np.ndarray | Sequence[Sequence[float]],
    reference_point: np.ndarray | Sequence[float],
) -> float:
    """Calculate the hypervolume indicator for a set of points.

    The hypervolume is the volume of the objective space dominated by the
    provided points, bounded by the reference point. All objectives are assumed
    to be minimization problems (lower is better).

    Parameters
    ----------
    points : np.ndarray | Sequence[Sequence[float]]
        Array of shape (n_points, n_objectives) representing the objective
        values for each solution. Each row is a point in the objective space.
    reference_point : np.ndarray | Sequence[float]
        Reference point of shape (n_objectives,) used to bound the hypervolume.
        Should be worse than all points (i.e., all points should dominate it).

    Returns
    -------
    float
        The hypervolume value. Higher values indicate better solution sets.

    Examples
    --------
    >>> import numpy as np
    >>> points = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]])
    >>> reference = np.array([3.0, 3.0])
    >>> hv = calculate_hypervolume(points, reference)
    >>> hv > 0
    True

    Notes
    -----
    - All objectives are assumed to be minimization (lower is better).
    - Points that are dominated by the reference point are filtered out.
    - The implementation uses a recursive algorithm for exact hypervolume
      computation in 2D and 3D, and an approximation for higher dimensions.
    """
    points = np.asarray(points, dtype=np.float64)
    reference_point = np.asarray(reference_point, dtype=np.float64)

    if points.ndim != 2:  # type: ignore[union-attr]
        raise ValueError(f"points must be 2D array, got shape {points.shape}")  # type: ignore[union-attr]
    if reference_point.ndim != 1:  # type: ignore[union-attr]
        raise ValueError(f"reference_point must be 1D array, got shape {reference_point.shape}")  # type: ignore[union-attr]
    if points.shape[1] != reference_point.shape[0]:  # type: ignore[union-attr]
        raise ValueError(
            f"points and reference_point must have same number of objectives: "
            f"{points.shape[1]} != {reference_point.shape[0]}"  # type: ignore[union-attr]
        )

    n_points, n_objectives = points.shape  # type: ignore[union-attr]

    if n_points == 0:
        return 0.0

    # Filter out points that are dominated by the reference point
    # (i.e., points where at least one objective is worse than reference)
    dominated_mask = np.all(points <= reference_point, axis=1)  # type: ignore[operator]
    if not np.any(dominated_mask):
        return 0.0

    filtered_points = points[dominated_mask]

    # Normalize points relative to reference point
    normalized_points = reference_point - filtered_points  # type: ignore[operator]

    # Ensure all values are positive (points should dominate reference)
    if np.any(normalized_points <= 0):
        raise ValueError(
            "All points must dominate the reference point. "
            "Ensure reference point is worse than all points in all objectives."
        )

    # Use appropriate algorithm based on dimensionality
    if n_objectives == 1:
        # 1D: hypervolume is the maximum distance from reference
        return float(np.max(normalized_points))
    elif n_objectives == 2:
        # 2D: use efficient 2D hypervolume algorithm
        return _hypervolume_2d(normalized_points)
    elif n_objectives == 3:
        # 3D: use efficient 3D hypervolume algorithm
        return _hypervolume_3d(normalized_points)
    else:
        # Higher dimensions: use recursive algorithm (slower but exact)
        return _hypervolume_nd(normalized_points)


def _hypervolume_2d(points: np.ndarray) -> float:
    """Calculate hypervolume for 2D case using efficient algorithm.

    Parameters
    ----------
    points : np.ndarray
        Normalized points of shape (n_points, 2) with all values > 0.

    Returns
    -------
    float
        Hypervolume value.
    """
    if len(points) == 0:
        return 0.0

    # Sort points by first objective (ascending)
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]

    # Calculate hypervolume using the "sweep line" approach
    hv = 0.0
    max_y = 0.0

    for i in range(len(sorted_points)):
        x_prev = sorted_points[i - 1, 0] if i > 0 else 0.0
        x_curr = sorted_points[i, 0]
        y_curr = sorted_points[i, 1]

        # Add rectangle: (x_curr - x_prev) * max_y
        hv += (x_curr - x_prev) * max_y

        # Update max_y for next iteration
        max_y = max(max_y, y_curr)

    # Add final rectangle to the end
    hv += sorted_points[-1, 0] * max_y

    return float(hv)


def _hypervolume_3d(points: np.ndarray) -> float:
    """Calculate hypervolume for 3D case using efficient algorithm.

    Parameters
    ----------
    points : np.ndarray
        Normalized points of shape (n_points, 3) with all values > 0.

    Returns
    -------
    float
        Hypervolume value.
    """
    if len(points) == 0:
        return 0.0

    # Sort points by first objective (ascending)
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]

    # Use "sweep plane" approach: integrate 2D hypervolumes along first axis
    hv = 0.0
    x_prev = 0.0

    for i in range(len(sorted_points)):
        x_curr = sorted_points[i, 0]

        # Get all points up to current x value, project to 2D (y, z)
        points_up_to_x = sorted_points[: i + 1, 1:]
        hv_2d = _hypervolume_2d(points_up_to_x)

        # Add volume slice
        hv += (x_curr - x_prev) * hv_2d
        x_prev = x_curr

    return float(hv)


def _hypervolume_nd(points: np.ndarray) -> float:
    """Calculate hypervolume for N-dimensional case using recursive algorithm.

    This is a slower but exact algorithm for higher dimensions.

    Parameters
    ----------
    points : np.ndarray
        Normalized points of shape (n_points, n_objectives) with all values > 0.

    Returns
    -------
    float
        Hypervolume value.
    """
    if len(points) == 0:
        return 0.0

    n_objectives = points.shape[1]

    if n_objectives == 1:
        return float(np.max(points))
    elif n_objectives == 2:
        return _hypervolume_2d(points)
    elif n_objectives == 3:
        return _hypervolume_3d(points)

    # Recursive case: integrate over first dimension
    # Sort by first objective
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]

    hv = 0.0
    x_prev = 0.0

    for i in range(len(sorted_points)):
        x_curr = sorted_points[i, 0]

        # Get points up to current x, project to lower dimension
        points_up_to_x = sorted_points[: i + 1, 1:]
        if len(points_up_to_x) > 0:
            hv_lower = _hypervolume_nd(points_up_to_x)
            hv += (x_curr - x_prev) * hv_lower

        x_prev = x_curr

    return float(hv)
