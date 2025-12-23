import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

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

from experiments.offline_optimization.helpers import manual_docking_target_transform

OptimizationMetrics = dict[str, float]
RegressionMetrics = dict[str, float]


def get_optimization_threshold(target: str) -> float:
    """Returns the optimization threshold for the given target."""
    return DOCKING_THRESHOLDS[target]


def calculate_optimization_metrics(
    solutions: Sequence[str],
    objective_function_values: np.ndarray,
    threshold: float,
    out_dir: Path | None = None,
    optimization_round: int | None = None,
) -> OptimizationMetrics:
    """Calculates the optimization metrics for the given solutions and objective function values."""
    solutions = np.array(solutions, dtype=str)
    objective_function_values = np.array(objective_function_values, dtype=float)

    intdiv1 = calculate_intdiv1(solutions)
    ds_scores = objective_function_values[:, 0]
    sa_scores = objective_function_values[:, 1]
    qed_scores = objective_function_values[:, 2]
    hit_ratio_no_constraints = calculate_hit_ratio(
        objective_scores=ds_scores, objective_threshold=threshold
    )
    hit_ratio = calculate_hit_ratio(
        objective_scores=ds_scores,
        objective_threshold=threshold,
        qed_scores=qed_scores,
        sa_scores=sa_scores,
    )

    normalized_objective_function_values = manual_docking_target_transform(
        objective_function_values
    )
    normalized_docking_scores = normalized_objective_function_values[:, 0]
    normalized_sa_scores = normalized_objective_function_values[:, 1]
    normalized_qed_scores = normalized_objective_function_values[:, 2]
    hv, r2 = calculate_hypervolume(
        normalized_docking_scores,
        normalized_sa_scores,
        normalized_qed_scores,
        solutions,
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


def calculate_regression_metrics(
    predicted: np.ndarray,
    true: np.ndarray,
    out_dir: Path | None = None,
    optimization_round: int | None = None,
) -> RegressionMetrics:
    """Calculates the regression metrics for the given predicted and true values.

    Calculates MAE and Spearman correlation for each target separately (DS, SA, QED).
    Expected input shape: (n_samples, 3) for both predicted and true arrays.
    """
    predicted = np.asarray(predicted)
    true = np.asarray(true)

    # Ensure 2D arrays
    if predicted.ndim == 1:
        predicted = predicted.reshape(-1, 1)
    if true.ndim == 1:
        true = true.reshape(-1, 1)

    # Get number of targets (should be 3: DS, SA, QED)
    num_targets = predicted.shape[1]
    target_names = ["ds", "sa", "qed"]

    current_metrics = {}

    # Calculate metrics for each target
    for target_idx in range(num_targets):
        target_name = (
            target_names[target_idx]
            if target_idx < len(target_names)
            else f"target_{target_idx}"
        )

        pred_target = predicted[:, target_idx]
        true_target = true[:, target_idx]

        # Calculate MAE
        mae = np.mean(np.abs(pred_target - true_target))

        # Calculate Spearman correlation
        spearman_corr = spearmanr(pred_target, true_target).correlation

        # Store metrics with target-specific keys
        current_metrics[f"{target_name}_mae"] = float(mae)
        current_metrics[f"{target_name}_spearman_corr"] = float(spearman_corr)

    # Also calculate overall metrics (averaged across targets)
    overall_mae = np.mean(np.abs(predicted - true))
    # For overall Spearman, we can calculate it on flattened arrays or average per-target correlations
    overall_spearman = np.mean(
        [spearmanr(predicted[:, i], true[:, i]).correlation for i in range(num_targets)]
    )
    current_metrics["overall_mae"] = float(overall_mae)
    current_metrics["overall_spearman_corr"] = float(overall_spearman)

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
