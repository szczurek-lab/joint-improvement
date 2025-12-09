"""Training script for Hyformer model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from loguru import logger

# Ensure experiments package is importable
if (project_root := Path(__file__).parent.parent.parent) not in [Path(p) for p in sys.path]:
    sys.path.insert(0, str(project_root))

from experiments.helpers import (
    load_collator,
    load_dataset,
    load_model,
    load_tokenizer,
    load_trainer,
)
from joint_improvement.utils import (
    LMCollator,
    MLMCollator,
    create_task_dataloaders,
    set_seed,
)


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

    # Set random seed
    set_seed(args.seed, deterministic=args.deterministic)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_dataset = load_dataset(args.train_dataset_config)
    val_dataset = load_dataset(args.val_dataset_config) if args.val_dataset_config else None

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_config, config_dump_dir=args.output_dir)

    # Load model
    model = load_model(model_config=args.model_config, device=args.device, config_dump_dir=args.output_dir)

    # Load trainer
    trainer = load_trainer(
        trainer_config=args.trainer_config,
        model=model,
        device=args.device,
        out_dir=args.output_dir,
        config_dump_dir=args.output_dir,
    )

    # Trainer must have tasks defined
    if not trainer.config.tasks:
        raise ValueError("Trainer config must have tasks defined. Cannot train without tasks.")

    # Trainer tasks must be a subset of tokenizer task tokens
    trainer_tasks = set(trainer.config.tasks.keys())
    tokenizer_tasks = set(tokenizer.task_tokens.keys())
    if not trainer_tasks.issubset(tokenizer_tasks):
        missing_tasks = trainer_tasks - tokenizer_tasks
        raise ValueError(
            f"Trainer tasks ({trainer_tasks}) must be a subset of tokenizer task tokens ({tokenizer_tasks}). "
            f"Missing task tokens: {missing_tasks}"
        )

    # check compatibility between tokenizer and model vocab size
    if model.vocab_size != len(tokenizer.vocab):
        raise ValueError(
            f"Model vocab size ({model.vocab_size}) != tokenizer vocab size "
            f"({len(tokenizer.vocab)}). They must match for training."
        )

    # So now it is tricky, I could in principle support passing datasets to Trainer
    # and then it will take care of the dataloaders

    def make_collator(task: str) -> LMCollator | MLMCollator:
        """Factory function to create collators for each task."""
        return load_collator(
            task=task,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )

    train_loaders, val_loaders = create_task_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tasks=trainer.config.tasks,
        collator_factory=make_collator,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    # Start training
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    logger.info(f"Device: {args.device}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Model parameters: {model.get_num_params(trainable_only=True):,}")
    logger.info(f"Tasks and weights: {trainer.config.tasks}")

    trainer.train(
        train_loaders=train_loaders,
        val_loaders=val_loaders,
    )

    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
