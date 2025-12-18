import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from experiments.helpers import load_dataset
from joint_improvement.utils import SequenceDataset
from joint_improvement.utils.chemistry.docking import DOCKING_THRESHOLDS


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

def calculate_model_predictions(
    solutions: Sequence[str],
    tokenizer: SMILESTokenizer,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Calculates the model predictions for the given solutions."""
    from joint_improvement.utils import SequenceDataLoader, PredictionCollator, SequenceDataset
    
    dataset = SequenceDataset(sequences=solutions)
    dataloader = SequenceDataLoader(
        dataset=dataset,
        collator=PredictionCollator(tokenizer=tokenizer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    for batch in dataloader:
    
    predictions = model.generate(solutions, device=device)
    return predictions