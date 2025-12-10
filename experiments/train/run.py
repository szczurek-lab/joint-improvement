"""Training script for Hyformer model."""

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
from joint_improvement.utils import SequenceDataLoader, set_seed  # noqa: E402


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
        "--trainer-config",
        type=Path,
        required=True,
        help="Path to TrainerConfig JSON file. If not provided, uses default values.",
    )
    parser.add_argument(
        "--model-ckpt",
        type=Path,
        help="Path to model checkpoint to load.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Path to trainer checkpoint to resume training from.",
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
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode for PyTorch (may reduce performance).",
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
        help="Batch size for training (default: 8).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization (default: 512).",
    )
    return parser


def main() -> None:
    """Main training function."""
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)
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
