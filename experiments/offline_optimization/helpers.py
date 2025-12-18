import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from experiments.helpers import load_dataset
from joint_improvement.utils import SequenceDataset
from joint_improvement.utils.chemistry import (
    calculate_qed_batch,
    calculate_sa_batch,
)
from joint_improvement.utils.chemistry.docking import DOCKING_THRESHOLDS
from joint_improvement.utils.metrics import (
    calculate_hit_ratio,
    calculate_hypervolume,
    calculate_intdiv1,
)

OptimizationMetrics = dict[str, float]
RegressionMetrics = dict[str, float]


def docking_target_transform(target: np.ndarray) -> np.ndarray:
    """Transform the docking targets to the desired range [0, 1]."""
    return np.array(
        [
            -target[0] / 20,  # docking score
            (10 - target[1]) / 9,  # sa
            target[2],  # qed
        ],
        dtype=np.float32,
    )


def inverse_docking_target_transform(target: np.ndarray) -> np.ndarray:
    """Inverse transform to convert normalized docking targets back to original scale."""
    return np.array(
        [
            -target[0] * 20,  # docking score: inverse of -x/20
            10 - target[1] * 9,  # sa: inverse of (10-x)/9
            target[2],  # qed: identity transform
        ],
        dtype=np.float32,
    )


def load_docking_dataset(
    dataset_config_path: Path, config_dump_dir: Path | None = None
) -> SequenceDataset:
    """Load a single docking dataset from configuration file and apply the docking target transform."""
    dataset = load_dataset(dataset_config_path, config_dump_dir)
    dataset.target_transforms = [docking_target_transform]
    return dataset


def get_optimization_threshold(target: str) -> float:
    """Returns the optimization threshold for the given target."""
    return DOCKING_THRESHOLDS[target]




def calculate_optimization_metrics(
    smiles: Sequence[str],
    objective_scores: Sequence[float],
    threshold: float,
    out_dir: Path | None = None,
    optimization_round: int | None = None,
) -> OptimizationMetrics:
    """Calculates the optimization metrics for the given smiles and objective scores."""
    smiles = np.array(smiles, dtype=str)
    objective_scores = np.array(objective_scores, dtype=float)

    intdiv1 = calculate_intdiv1(smiles)
    qed_scores = calculate_qed_batch(smiles)
    sa_scores = calculate_sa_batch(smiles)
    hit_ratio_no_constraints = calculate_hit_ratio(
        objective_scores=objective_scores, objective_threshold=threshold
    )
    hit_ratio = calculate_hit_ratio(
        objective_scores=objective_scores,
        objective_threshold=threshold,
        qed_scores=qed_scores,
        sa_scores=sa_scores,
    )
    normalized_docking_scores = -objective_scores / 20  # type: ignore[operator]
    normalized_sa_scores = (10 - sa_scores) / 9
    normalized_qed_scores = qed_scores
    hv, r2 = calculate_hypervolume(
        normalized_docking_scores, normalized_qed_scores, normalized_sa_scores, smiles
    )
    current_metrics = {
        "intdiv1": float(intdiv1),
        "hit_ratio_no_constraints": float(hit_ratio_no_constraints),
        "hit_ratio": float(hit_ratio),
        "hv": float(hv),
        "r2": float(r2),
    }

    if out_dir is not None:
        metrics_file = out_dir / "optimization_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                all_metrics = json.load(f)
            if optimization_round is not None:
                all_metrics[f"round_{optimization_round}"] = current_metrics
            else:
                all_metrics.update(current_metrics)
        else:
            if optimization_round is not None:
                all_metrics = {f"round_{optimization_round}": current_metrics}
            else:
                all_metrics = current_metrics
        with open(metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=2)

    return current_metrics


def save_solutions(
    smiles: Sequence[str],
    objective_values: Sequence[Sequence[float]],
    out_dir: Path | None = None,
    optimization_round: int | None = None,
) -> None:
    """Saves SMILES strings with their corresponding objective values to a JSON file.

    Parameters
    ----------
    smiles : Sequence[str]
        List of SMILES strings.
    objective_values : Sequence[Sequence[float]]
        List of objective value arrays, where each array contains [docking_score, sa, qed].
    out_dir : Path | None, optional
        Output directory to save the solutions file. If None, no file is saved.
    optimization_round : int | None, optional
        Optimization round number. If provided, solutions are stored per round.
    """
    if out_dir is None:
        return

    # Convert to lists and ensure they're the same length
    smiles_list = list(smiles)
    objective_values_list = [list(obj_vals) for obj_vals in objective_values]

    if len(smiles_list) != len(objective_values_list):
        raise ValueError(
            f"Length mismatch: {len(smiles_list)} SMILES but {len(objective_values_list)} objective values"
        )

    # Validate that each objective value has exactly 3 components
    for i, obj_vals in enumerate(objective_values_list):
        if len(obj_vals) != 3:
            raise ValueError(
                f"Expected 3 objective values (docking_score, sa, qed) for SMILES {i}, "
                f"but got {len(obj_vals)} values"
            )

    # Create current solutions data with explicit breakdown
    current_solutions = [
        {
            "smiles": smi,
            "docking_score": float(obj_vals[0]),
            "sa": float(obj_vals[1]),
            "qed": float(obj_vals[2]),
        }
        for smi, obj_vals in zip(smiles_list, objective_values_list, strict=True)
    ]

    solutions_file = out_dir / "solutions.json"
    if solutions_file.exists():
        with open(solutions_file) as f:
            all_solutions = json.load(f)
        if optimization_round is not None:
            all_solutions[f"round_{optimization_round}"] = current_solutions
        else:
            # If no round specified, append to existing list or update
            if isinstance(all_solutions, list):
                all_solutions.extend(current_solutions)
            else:
                all_solutions["solutions"] = current_solutions
    else:
        if optimization_round is not None:
            all_solutions = {f"round_{optimization_round}": current_solutions}
        else:
            all_solutions = current_solutions

    with open(solutions_file, "w") as f:
        json.dump(all_solutions, f, indent=2)


def calculate_regression_metrics(
    predicted: Sequence[float],
    true: Sequence[float],
    inverse_transform: Callable[[np.ndarray], np.ndarray],
    out_dir: Path | None = None,
    optimization_round: int | None = None,
) -> RegressionMetrics:
    """Calculates the regression metrics for the given predicted and true values."""
    predicted_arr = inverse_transform(np.array(predicted, dtype=np.float32))
    true_arr = inverse_transform(np.array(true, dtype=np.float32))
    mae = np.mean(np.abs(predicted_arr - true_arr))
    spearman_corr = spearmanr(predicted_arr, true_arr).correlation
    current_metrics = {"mae": float(mae), "spearman_corr": float(spearman_corr)}

    if out_dir is not None:
        metrics_file = out_dir / "regression_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                all_metrics = json.load(f)
            if optimization_round is not None:
                all_metrics[f"round_{optimization_round}"] = current_metrics
            else:
                all_metrics.update(current_metrics)
        else:
            if optimization_round is not None:
                all_metrics = {f"round_{optimization_round}": current_metrics}
            else:
                all_metrics = current_metrics
        with open(metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=2)

    return current_metrics

