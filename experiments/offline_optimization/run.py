"""Offline optimization script for Joint Self-Improvement."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from joint_improvement.utils.chemistry import (
    calculate_docking_batch,
    calculate_qed_batch,
    calculate_sa_batch,
)

# Ensure experiments package is importable
if (project_root := Path(__file__).parent.parent.parent) not in [
    Path(p) for p in sys.path
]:
    sys.path.insert(0, str(project_root))

from experiments.helpers import create_dataloader, load_model, load_tokenizer, load_trainer  # noqa: E402
from experiments.offline_optimization.helpers import (  # noqa: E402
    calculate_optimization_metrics,
    calculate_regression_metrics,
    get_optimization_threshold,
    inverse_docking_target_transform,
    load_docking_dataset,
    save_solutions,
)
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
    return parser


def oracle_call(solutions: list[str], target: str) -> np.ndarray:
    ds = calculate_docking_batch(solutions, target=target)
    sa = calculate_sa_batch(solutions)
    qed = calculate_qed_batch(solutions)
    return np.concatenate([ds.reshape(-1, 1), sa.reshape(-1, 1), qed.reshape(-1, 1)], axis=1)


def main() -> None:
    """Performs offline optimization by calling the oracle and updating the offline dataset."""
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed, deterministic=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    solutions: list[str] = []
    objective_values = []
    docking_scores = []
    threshold = get_optimization_threshold(args.target)

    # Load offline dataset and test dataset once before the optimization loop
    offline_dataset = load_docking_dataset(
        args.train_dataset_config, config_dump_dir=args.output_dir
    )
    test_dataset = load_docking_dataset(
        args.test_dataset_config, config_dump_dir=None  # no need to save test dataset config
    )

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
    while len(offline_dataset) < args.oracle_budget:

        # 0. Log optimization round
        logger.info("-" * 100)
        logger.info(f"Optimization round {optimization_round} started...")
        logger.info("-" * 100)

        # 1. Load pretrained tokenizer/model

        trainer = load_trainer(
            args.trainer_config,
            model,
            args.device,
            args.output_dir,
            config_dump_dir=args.output_dir,
        )

        # 2. Jointly train model on offline dataset
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
        model.train()
        trainer.train(train_loaders=train_loaders, val_loaders=val_loaders)

        # 3. Report test set predictive performance
        outputs = trainer.test(dataloader=val_loaders["prediction"])
        y_true = outputs["true"][:, 0].detach().cpu().numpy()
        y_pred = outputs["predicted"][:, 0].detach().cpu().numpy()
        regression_metrics = calculate_regression_metrics(
            y_pred,
            y_true,
            inverse_docking_target_transform,
            out_dir=args.output_dir,
            optimization_round=optimization_round,
        )
        logger.info(
            f"Optimization round {optimization_round} regression metrics: {regression_metrics}"
        )

        # 4. Sample new solutions
        model.eval()
        oracle = partial(oracle_fn, tokenizer=tokenizer, target=args.target, model=model, model_device=args.device)
        _sampled_solutions = sample_new_solutions(
            tokenizer=tokenizer,
            model=model,
            advantage_fn=lambda x: x,
            oracle=oracle,
            device=args.device,
            num_samples=args.batch_size,
            max_sequence_length=args.max_length,
            temperature=1.0,
            top_k=25,
            beam_width=16,
            num_rounds=10,
            advantage_constant=1.0,
            normalize_advantage_value=True,
            min_nucleus_top_p=0.95,
        )
        logger.info(f"Sampled {len(_sampled_solutions)} solutions")

        # 5. Oracle call and update offline dataset
        _sampled_objective_values = oracle_call(list(_sampled_solutions), target=args.target)
        solutions.extend(_sampled_solutions)
        docking_scores.extend(_sampled_objective_values[:, 0].tolist())
        objective_values.extend(_sampled_objective_values.tolist())

        # 6. Calculate optimization metrics
        optimization_metrics = calculate_optimization_metrics(
            solutions,
            docking_scores,
            threshold,
            out_dir=args.output_dir,
            optimization_round=optimization_round,
        )
        logger.info(f"Optimization round {optimization_round} optimization metrics: {optimization_metrics}")

        # 7. Save solutions and objective values
        save_solutions(
            solutions,
            objective_values,
            out_dir=args.output_dir,
            optimization_round=optimization_round,
        )

        # 8. Add new solutions to the offline dataset
        offline_dataset.append(sequences=_sampled_solutions, targets=_sampled_objective_values.tolist())

        # 9. Increment optimization round counter
        optimization_round += 1


if __name__ == "__main__":
    main()
