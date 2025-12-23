"""Offline optimization script for Joint Self-Improvement."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from loguru import logger

# Ensure experiments package is importable
if (project_root := Path(__file__).parent.parent.parent) not in [
    Path(p) for p in sys.path
]:
    sys.path.insert(0, str(project_root))

from experiments.helpers import create_dataloader, load_model, load_tokenizer, load_trainer  # noqa: E402
from experiments.offline_optimization.helpers import (  # noqa: E402
    load_docking_datasets,
    save_solutions,
)
from experiments.offline_optimization.metrics import (  # noqa: E402
    calculate_optimization_metrics,
    calculate_regression_metrics,
    get_optimization_threshold,
)
from experiments.offline_optimization.oracle import oracle_call  # noqa: E402
from experiments.offline_optimization.sample import oracle_fn, sample_new_solutions  # noqa: E402
from joint_improvement.utils import SequenceDataLoader, set_seed  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Offline optimization script for Joint Self-Improvement.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save optimization results.",
    )
    parser.add_argument(
        "--train-dataset-config",
        type=Path,
        required=True,
        help="Path to SequenceDatasetConfig JSON file for offline dataset.",
    )
    parser.add_argument(
        "--test-dataset-config",
        type=Path,
        required=True,
        help="Path to SequenceDatasetConfig JSON file for test dataset.",
    )
    parser.add_argument(
        "--tokenizer-config",
        type=Path,
        required=True,
        help="Path to tokenizer directory containing tokenizer_config.json and vocabulary file.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Path to model configuration JSON file.",
    )
    parser.add_argument(
        "--model-ckpt",
        type=Path,
        required=False,
        default=None,
        help="Path to model checkpoint to load. If not provided, defaults to None.",
    )
    parser.add_argument(
        "--trainer-config",
        type=Path,
        required=True,
        help="Path to optimization configuration JSON file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to optimize on (default: cuda if available, else cpu).",
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
        help="Number of workers to use for parallel processing (default: 4).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for each optimization round (default: 16).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization (default: 128).",
    )
    parser.add_argument(
        "--oracle-budget",
        type=int,
        default=100,
        help="Budget for the oracle calls (default: 100).",
    )
    parser.add_argument(
        "--augmentation-rounds",
        type=int,
        default=0,
        help="Number of augmentation rounds (default: 0).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling new solutions (default: 1.0).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Top-k for sampling new solutions (default: 25).",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=16,
        help="Beam width for sampling new solutions (default: 16).",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=10,
        help="Number of rounds for sampling new solutions (default: 10).",
    )
    parser.add_argument(
        "--advantage-constant",
        type=float,
        default=1.0,
        help="Advantage constant for sampling new solutions (default: 1.0).",
    )
    parser.add_argument(
        "--normalize-advantage-value",
        type=bool,
        default=True,
        help="Normalize advantage value for sampling new solutions (default: True).",
    )
    parser.add_argument(
        "--min-nucleus-top-p",
        type=float,
        default=0.95,
        help="Minimum nucleus top-p for sampling new solutions (default: 0.95).",
    )
    return parser


def main() -> None:
    """Performs offline optimization by calling the oracle and updating the offline dataset."""
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed, deterministic=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    solutions: list[str] = []
    threshold = get_optimization_threshold(args.target)

    # Load offline dataset, tokenizer and model
    offline_dataset, test_dataset = load_docking_datasets(
        args.train_dataset_config, args.test_dataset_config, config_dump_dir=args.output_dir
    )
    logger.info(f"Loaded {len(offline_dataset)} solutions from offline dataset")
    
    tokenizer = load_tokenizer(
        args.tokenizer_config, config_dump_dir=args.output_dir
    )

    model = load_model(
        args.model_config,
        args.model_ckpt,
        args.device,
        config_dump_dir=args.output_dir,
    )

    optimization_round: int = 0
    while True:
        
        if len(solutions) >= args.oracle_budget:
            break

        logger.info("-" * 100)
        logger.info(f"Optimization round {optimization_round} started...")
        logger.info("-" * 100)

        # 1. Jointly update the model using the offline dataset
        trainer = load_trainer(
            args.trainer_config,
            model,
            args.device,
            args.output_dir,
            config_dump_dir=args.output_dir,
        )

        train_loaders: dict[str, SequenceDataLoader] = {}
        val_loaders: dict[str, SequenceDataLoader] = {}

        for task_name in trainer.config.tasks.keys():
            train_loaders[task_name] = create_dataloader(
                dataset=offline_dataset,
                tokenizer=tokenizer,
                task_name=task_name,
                batch_size=trainer.config.batch_size,
                num_workers=args.num_workers,
                device=args.device,
                shuffle=True,
            )
            val_loaders[task_name] = create_dataloader(
                dataset=test_dataset,
                tokenizer=tokenizer,
                task_name=task_name,
                batch_size=trainer.config.batch_size,
                num_workers=args.num_workers,
                device=args.device,
                shuffle=False,
            )
        # trainer.train() will set model to train mode internally
        trainer.train(train_loaders=train_loaders, val_loaders=val_loaders)

        # 3. Report the test set predictive performance for all targets (DS, SA, QED)
        outputs = trainer.test(dataloader=val_loaders["prediction"])
        for target_transform in test_dataset.target_transforms:
            y_true = target_transform.inverse_transform(outputs["true"].detach().cpu().numpy())
            y_pred = target_transform.inverse_transform(outputs["predicted"].detach().cpu().numpy())
        regression_metrics = calculate_regression_metrics(
            y_pred,
            y_true,
            out_dir=args.output_dir,
            optimization_round=optimization_round,
        )
        logger.info(
            f"Optimization round {optimization_round} regression metrics: {regression_metrics}"
        )

        # 4. Sample new solutions from the model
        # Use underlying_model for sampling to ensure consistency (compiled models may have issues with generation)
        # The generator will automatically set the model to eval mode during sampling
        sampling_model = trainer.underlying_model
        oracle = partial(
            oracle_fn,
            tokenizer=tokenizer,
            target_transforms=offline_dataset.target_transforms,
            model=sampling_model,
            model_device=args.device,
        )
        _sampled_solutions = sample_new_solutions(
            tokenizer=tokenizer,
            model=sampling_model,
            advantage_fn=lambda x: x, # try x ** 3
            oracle=oracle,
            device=args.device,
            num_samples=args.batch_size,
            max_sequence_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            beam_width=args.beam_width,
            num_rounds=args.num_rounds,
            advantage_constant=args.advantage_constant,
            normalize_advantage_value=args.normalize_advantage_value,
            min_nucleus_top_p=args.min_nucleus_top_p,
        )
        logger.info(f"Sampled {len(_sampled_solutions)} solutions")
        solutions.extend(_sampled_solutions)
        solutions = list(set(solutions))
        
        # 5. Add new solutions to the offline dataset
        #offline_dataset.append(sequences=_sampled_solutions, targets=_sampled_objective_values.tolist())

        # 6. Increment optimization round counter
        #optimization_round += 1
    
    # Evaluate the solutions using the oracle
    if len(solutions) > args.oracle_budget:
        solutions = solutions[-args.oracle_budget:]
        assert len(solutions) == args.oracle_budget
        
    objective_function_values = oracle_call(solutions, target=args.target)
    optimization_metrics = calculate_optimization_metrics(
        solutions,
        objective_function_values,
        threshold,
        out_dir=args.output_dir
    )
    
    logger.info(f"Optimization metrics for offline optimization: {optimization_metrics}")
    save_solutions(
        solutions,
        objective_function_values,
        out_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
