"""Oracle functions for offline optimization."""

from __future__ import annotations

import numpy as np

from joint_improvement.utils.chemistry import (
    calculate_docking_batch,
    calculate_qed_batch,
    calculate_sa_batch,
)


def oracle_call(solutions: list[str], target: str) -> np.ndarray:
    """Call the oracle to evaluate solutions.
    
    Parameters
    ----------
    solutions : list[str]
        List of SMILES strings to evaluate.
    target : str
        Target protein for docking calculations.
    
    Returns
    -------
    np.ndarray
        Array of shape (n_solutions, 3) containing [docking_score, sa, qed] for each solution.
    """
    ds = calculate_docking_batch(solutions, target=target)
    sa = calculate_sa_batch(solutions)
    qed = calculate_qed_batch(solutions)
    return np.concatenate([ds.reshape(-1, 1), sa.reshape(-1, 1), qed.reshape(-1, 1)], axis=1)
