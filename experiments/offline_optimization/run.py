"""Offline optimization script for Joint Self-Improvement."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import torch

# Suppress dynamo warnings about non-relation expressions
# These warnings occur when PyTorch's compiler encounters complex symbolic shape
# expressions (e.g., OR conditions) that it cannot fully optimize. They are harmless
# and don't affect training correctness, but can clutter the output.
warnings.filterwarnings(
    "ignore",
    message=".*_maybe_guard_rel.*was called on non-relation expression.*",
    category=UserWarning,
)

# Ensure experiments package is importable
if (project_root := Path(__file__).parent.parent.parent) not in [Path(p) for p in sys.path]:
    sys.path.insert(0, str(project_root))

from experiments.helpers import create_dataloader, load_dataset, load_model, load_tokenizer, load_trainer  # noqa: E402
from joint_improvement.hyformer.model import Hyformer  # noqa: E402
from joint_improvement.tokenizers.smiles import SMILESTokenizer  # noqa: E402
from joint_improvement.utils import SequenceDataLoader, SequenceDataset, set_seed  # noqa: E402
from joint_improvement.utils.chemistry.docking import (  # noqa: E402
    TARGET_DOCKING_THRESHOLDS,
    calculate_docking_batch,
)
from joint_improvement.utils.chemistry.qed import calculate_qed_batch  # noqa: E402
from joint_improvement.utils.chemistry.sa import calculate_sa_batch  # noqa: E402
from joint_improvement.utils.metrics.docking.hit_ratio import calculate_hit_ratio  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Train Hyformer model in multi-task setting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save checkpoints and logs.",
    )
    parser.add_argument(
        "--train-dataset-config",
        type=Path,
        required=True,
        help="Path to SequenceDatasetConfig JSON file for training data.",
    )
    parser.add_argument(
        "--val-dataset-config",
        type=Path,
        required=True,
        help="Path to SequenceDatasetConfig JSON file for validation data.",
    )
    parser.add_argument(
        "--tokenizer-config",
        type=Path,
        required=True,
        help="Path to tokenizer directory containing tokenizer_config.json and vocab file.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Path to HyformerConfig JSON file.",
    )
    parser.add_argument(
        "--model-ckpt",
        type=Path,
        required=True,
        help="Path to model checkpoint to load.",
    )
    parser.add_argument(
        "--trainer-config",
        type=Path,
        required=True,
        help="Path to TrainerConfig JSON file. If not provided, uses default values.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0).",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target to optimize for.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers to use for data loading (default: 4).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for optimization (default: 8).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for optimization (default: 512).",
    )
    parser.add_argument(
        "--number_of_optimization_rounds",
        type=int,
        default=10,
        help="Number of optimization rounds (default: 10).",
    )
    parser.add_argument(
        "--number_of_optimization_steps",
        type=int,
        default=10,
        help="Number of optimization steps per round (default: 10).",
    )
    return parser


def optimization_round(
    model: Hyformer,
    offline_dataset: SequenceDataset,
    test_dataset: SequenceDataset,
    tokenizer: SMILESTokenizer,
    args: argparse.Namespace,
) -> None:
    return None


def reward_function(generated_molecules: list[str], target: str, device: str = "cpu") -> float:
    """Compute hit ratio-based reward for generated molecules."""
    docking_scores = calculate_docking_batch(generated_molecules, target=target, device=device)
    qed_scores = calculate_qed_batch(generated_molecules)
    sa_scores = calculate_sa_batch(generated_molecules)
    docking_threshold = TARGET_DOCKING_THRESHOLDS.get(target, -9.0)

    return calculate_hit_ratio(
        docking_scores=docking_scores,
        qed_scores=qed_scores,
        sa_scores=sa_scores,
        docking_threshold=docking_threshold,
        qed_threshold=0.5,
        sa_threshold=5.0,
    )


def main() -> None:
    """Main offline optimization function.

    1. Load offline dataset
    2. Jointly fine-tune model on offline dataset
    3. Evaluate the model on test dataset and generation metrics
    4. Reason and Self-Improve the generated molecules
    5. Augment the offline dataset with the generated molecules
    6. Repeat
    """
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed, deterministic=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = load_dataset(args.train_dataset_config)
    val_dataset = load_dataset(args.val_dataset_config)

    tokenizer = load_tokenizer(args.tokenizer_config, config_dump_dir=args.output_dir)
    model = load_model(args.model_config, args.model_ckpt, args.device, config_dump_dir=args.output_dir)
    trainer = load_trainer(args.trainer_config, model, args.device, args.output_dir, config_dump_dir=args.output_dir)

    train_loaders: dict[str, SequenceDataLoader] = {}
    val_loaders: dict[str, SequenceDataLoader] = {}

    for task_name in trainer.config.tasks.keys():
        train_loaders[task_name] = create_dataloader(
            dataset=train_dataset,
            tokenizer=tokenizer,
            task_name=task_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            shuffle=True,
        )
        val_loaders[task_name] = create_dataloader(
            dataset=val_dataset,
            tokenizer=tokenizer,
            task_name=task_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            shuffle=False,
        )

    trainer.train(train_loaders=train_loaders, val_loaders=val_loaders)


if __name__ == "__main__":
    main()
