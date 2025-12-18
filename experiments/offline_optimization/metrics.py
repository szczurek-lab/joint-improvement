import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from joint_improvement.utils.chemistry import (
    calculate_qed_batch,
    calculate_sa_batch,
)
from joint_improvement.utils.metrics import (
    calculate_hit_ratio,
    calculate_hypervolume,
    calculate_intdiv1,
)

OptimizationMetrics = dict[str, float]
RegressionMetrics = dict[str, float]


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
    normalized_docking_scores = -ds_scores / 20
    normalized_sa_scores = (10 - sa_scores) / 9
    normalized_qed_scores = qed_scores
    hv, r2 = calculate_hypervolume(
        normalized_docking_scores, normalized_qed_scores, normalized_sa_scores, solutions
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
    spearman_corr = spearmanr(predicted_arr, true_arr).statistic
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
