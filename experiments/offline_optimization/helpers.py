import json
from collections.abc import Sequence
from pathlib import Path

import torch
import numpy as np
from loguru import logger

from experiments.helpers import load_dataset
from joint_improvement.tokenizers import SMILESTokenizer
from joint_improvement.hyformer import Hyformer
from joint_improvement.utils import SequenceDataset, SequenceDataLoader, PredictionCollator
from joint_improvement.utils.chemistry.docking import DOCKING_THRESHOLDS
from joint_improvement.utils.data.target_transforms import ZScoreScaler


def manual_docking_target_transform(target: np.ndarray) -> np.ndarray:
    assert target.ndim == 2, f"Expected 2D array, but got {target.ndim}D array with shape {target.shape}"
    assert target.shape[1] == 3, f"Expected second dimension to be 3, but got shape {target.shape}"

    result = np.array(
        [
            -target[:, 0] / 20,  # docking score
            (10 - target[:, 1]) / 9,  # sa
            target[:, 2],  # qed
        ],
        dtype=target.dtype,
    ).T  
    
    return result


def load_docking_datasets(
    train_dataset_config_path: Path, test_dataset_config_path: Path, config_dump_dir: Path | None = None
) -> tuple[SequenceDataset, SequenceDataset]:
    """Load a single docking dataset from configuration file and apply the docking target transform."""
    # load train and test datasets
    train_dataset = load_dataset(train_dataset_config_path, config_dump_dir)
    test_dataset = load_dataset(test_dataset_config_path, config_dump_dir)
    
    # remove SMILES with invalid docking scores (target[0] > 0)
    for dataset in [train_dataset, test_dataset]:
        targets_array = np.array(dataset.targets)
        # Create mask for valid entries (docking score <= 0)
        valid_mask = targets_array[:, 0] <= 0
        # Filter sequences and targets
        filtered_sequences = [seq for seq, valid in zip(dataset.sequences, valid_mask) if valid]
        filtered_targets = targets_array[valid_mask].tolist()
        # Create new dataset with filtered data
        dataset._sequences = filtered_sequences
        dataset._targets = filtered_targets
        logger.info(f"Filtered dataset: removed {np.sum(~valid_mask)} entries with invalid docking scores, "
                   f"{len(filtered_sequences)} entries remaining")
    
    # get mean, std of all targets from train data
    mean = np.mean(train_dataset.targets, axis=0)
    std = np.std(train_dataset.targets, axis=0)
    logger.info(f"Inferred train dataset mean: {mean}, std: {std}")
    
    # Use ZScoreScaler directly
    train_dataset.target_transforms = [ZScoreScaler(mean, std)]
    test_dataset.target_transforms = [ZScoreScaler(mean, std)]
    return train_dataset, test_dataset


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
    model: Hyformer,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Calculates the model predictions for the given solutions."""
    dataset = SequenceDataset(sequences=solutions)
    dataloader = SequenceDataLoader(
        dataset=dataset,
        collator=PredictionCollator(tokenizer=tokenizer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    was_training = model.training
    model.eval()

    logits_chunks: list[torch.Tensor] = []
    for batch in tqdm(dataloader, desc="Running model predictions", leave=False):
        batch_device = batch.to(device)
        outputs = model(
            input_ids=batch_device.input_ids,
            attention_mask=batch_device.attention_mask,
            task=batch_device.task,
        )
    logits_chunks.append(outputs.logits.detach().float().to(device="cpu"))

    if was_training:
        model.train()

    predicted = torch.cat(logits_chunks, dim=0)
    return predicted
