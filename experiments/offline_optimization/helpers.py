import gc
import json
from collections.abc import Sequence
from pathlib import Path

import torch
import numpy as np
from loguru import logger

from experiments.helpers import load_dataset
from joint_improvement.tokenizers import SMILESTokenizer
from joint_improvement.hyformer import Hyformer
from joint_improvement.utils import (
    SequenceDataset,
    SequenceDataLoader,
    PredictionCollator,
    LMCollator,
)
from joint_improvement.utils.chemistry.docking import DOCKING_THRESHOLDS
from joint_improvement.utils.data.target_transforms import ZScoreScaler
from joint_improvement.utils.memory import free_gpu_memory


def manual_docking_target_transform(target: np.ndarray) -> np.ndarray:
    assert (
        target.ndim == 2
    ), f"Expected 2D array, but got {target.ndim}D array with shape {target.shape}"
    assert (
        target.shape[1] == 3
    ), f"Expected second dimension to be 3, but got shape {target.shape}"

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
    offline_dataset_config_path: Path,
    pretrain_dataset_config_path: Path | None = None,
    test_dataset_config_path: Path | None = None,
    config_dump_dir: Path | None = None,
) -> dict[str, SequenceDataset | None]:
    """Load a single docking dataset from configuration file and apply the docking target transform."""
    # load datasets
    datasets = {
        "offline": load_dataset(offline_dataset_config_path, config_dump_dir),
        "pretrain": (
            load_dataset(pretrain_dataset_config_path, None)
            if pretrain_dataset_config_path is not None
            else None
        ),
        "test": (
            load_dataset(test_dataset_config_path, None)
            if test_dataset_config_path is not None
            else None
        ),
    }

    # remove SMILES with invalid docking scores (target[0] > 0)
    for dataset in datasets.values():
        if dataset is None or dataset.targets is None:
            continue
        targets_array = np.array(dataset.targets)
        # Create mask for valid entries (docking score <= 0)
        valid_mask = targets_array[:, 0] < 0
        # Filter sequences and targets
        filtered_sequences = [
            seq for seq, valid in zip(dataset.sequences, valid_mask) if valid
        ]
        filtered_targets = targets_array[valid_mask].tolist()
        # Create new dataset with filtered data
        dataset._sequences = filtered_sequences
        dataset._targets = filtered_targets
        logger.info(
            f"Filtered dataset: removed {np.sum(~valid_mask)} entries with invalid docking scores, "
            f"{len(filtered_sequences)} entries remaining"
        )

    # get mean, std of all targets from train data
    mean = np.mean(datasets["offline"].targets, axis=0)
    std = np.std(datasets["offline"].targets, axis=0)
    logger.info(f"Inferred offline dataset mean: {mean}, std: {std}")

    # Use ZScoreScaler directly
    datasets["offline"].target_transforms = [ZScoreScaler(mean, std)]
    if datasets["test"] is not None:
        datasets["test"].target_transforms = [ZScoreScaler(mean, std)]
    return datasets


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


@torch.no_grad()
def calculate_model_predictions(
    solutions: Sequence[str],
    tokenizer: SMILESTokenizer,
    model: Hyformer,
    device: torch.device,
    batch_size: int = 64,
    target_transforms: list[ZScoreScaler] | None = None,
) -> np.ndarray:
    """Calculates the model predictions for the given solutions."""
    dataset = SequenceDataset(sequences=solutions)
    dataloader = SequenceDataLoader(
        dataset=dataset,
        collator=PredictionCollator(tokenizer=tokenizer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        drop_last=False,
    )
    was_training = model.training
    model.eval()

    logits_chunks: list[torch.Tensor] = []
    for batch in dataloader:
        batch_device = batch.to(device)
        outputs = model(
            input_ids=batch_device.input_ids,
            attention_mask=batch_device.attention_mask,
            task=batch_device.task,
        )
        logits = outputs.logits.detach().cpu().float()
        logits_chunks.append(logits)

    if was_training:
        model.train()

    predicted = torch.cat(logits_chunks, dim=0)
    if target_transforms is not None:
        for target_transform in target_transforms:
            predicted = target_transform.inverse_transform(predicted)

    return predicted


@torch.no_grad()
def calculate_model_perplexity(
    solutions: Sequence[str],
    tokenizer: SMILESTokenizer,
    model: Hyformer,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Calculates the model perplexity for the given solutions."""
    dataset = SequenceDataset(sequences=solutions)
    dataloader = SequenceDataLoader(
        dataset=dataset,
        collator=LMCollator(tokenizer=tokenizer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        drop_last=False,
    )
    was_training = model.training
    model.eval()

    logit_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    for batch in dataloader:
        batch_device = batch.to(device)
        outputs = model(
            input_ids=batch_device.input_ids,
            attention_mask=batch_device.attention_mask,
            task=batch_device.task,
        )
        logits = outputs.logits.detach().cpu().float()
        labels = batch_device.labels.detach().cpu()
        
        logit_batches.append(logits)
        label_batches.append(labels)

    if was_training:
        model.train()

    logits = torch.cat(logit_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)
    
    # Calculate perplexity from logits
    return _perplexity_from_logits(logits, labels)


def _perplexity_from_logits(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> np.ndarray:
    """Compute sequence-level perplexity from token logits.
    
    Based on the implementation from seqme (https://github.com/szczurek-lab/seqme).
    
    Parameters
    ----------
    logits : torch.Tensor
        Float tensor of shape (batch, seq_len, vocab_size) with unnormalized scores.
    labels : torch.Tensor
        Long tensor of shape (batch, seq_len) with token ids used as targets.
    ignore_index : int, optional
        Index to ignore in the labels. Default is -100.
        
    Returns
    -------
    np.ndarray
        Array of shape (batch,) with perplexity per sequence.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape (batch, seq_len, vocab_size), got {logits.shape}")
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape (batch, seq_len), got {labels.shape}")
    if labels.shape[:2] != logits.shape[:2]:
        raise ValueError(f"labels and logits must share (batch, seq_len), got {labels.shape} and {logits.shape}")
    
    # Log-softmax over the vocabulary for numerical stability
    log_probs = torch.log_softmax(logits, dim=-1)  # (batch, seq_len, vocab)
    
    # Shift log_probs and labels by one for next token prediction
    log_probs = log_probs[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    
    ppls = torch.zeros(log_probs.shape[0])
    for idx, (log_prob, label) in enumerate(zip(log_probs, labels, strict=True)):
        ppl = 0.0
        n = 0
        for lp, lab in zip(log_prob, label, strict=True):
            if lab == ignore_index:
                continue
            n += 1
            ppl += lp[lab].item()
        if n > 0:
            ppls[idx] = ppl / n
        else:
            ppls[idx] = float('inf')  # Handle empty sequence case
    
    ppls = torch.exp(-ppls)
    
    return ppls.cpu().numpy().astype(float)


def free_memory(trainer: object):
    free_gpu_memory(
        model=None,  # Keep model on GPU (not needed for docking, but keep in case)
        trainer=trainer,
        move_model_to_cpu=False,  # Set to True if you want to free even more memory
        clear_optimizer=True,  # This frees the most memory (~2x model size for Adam)
        clear_cache=True,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        