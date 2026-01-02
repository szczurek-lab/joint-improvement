"""Offline optimization script for Joint Self-Improvement.

Run offline optimization for
- N: number of self-improvement rounds
- B: batch size for each self-improvement round
- oracle-budget: number of solutions to sample from the oracle
"""

from __future__ import annotations

import os
import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

# Ensure experiments package is importable
if (project_root := Path(__file__).parent.parent.parent) not in [
    Path(p) for p in sys.path
]:
    sys.path.insert(0, str(project_root))

from experiments.helpers import (
    create_dataloader,
    load_model,
    load_tokenizer,
    load_trainer,
)  # noqa: E402
from experiments.offline_optimization.helpers import (  # noqa: E402
    load_docking_datasets,
    save_solutions,
    free_memory,
)
from experiments.offline_optimization.metrics import (  # noqa: E402
    calculate_optimization_metrics,
    calculate_regression_metrics,
    get_optimization_threshold,
)
from experiments.offline_optimization.oracle import oracle_call  # noqa: E402
from experiments.offline_optimization.sample import (
    oracle_fn,
    sample_solutions,
)  # noqa: E402
from joint_improvement.utils import SequenceDataLoader, set_seed  # noqa: E402
from joint_improvement.utils.chemistry.docking import DOCKING_THRESHOLDS  # noqa: E402


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
        "--offline-dataset-config",
        type=Path,
        required=True,
        help="Path to SequenceDatasetConfig JSON file for offline dataset.",
    )
    parser.add_argument(
        "--pretrain-dataset-config",
        type=Path,
        required=False,
        help="Path to SequenceDatasetConfig JSON file for pretraining dataset.",
        default=None,
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
        "--num-self-improvement-rounds",
        type=int,
        default=10,
        help="Number of self-improvement rounds (default: 10).",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=10,
        help="Number of rounds to sample from the model (default: 10).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for each self-improvement round (default: 64).",
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
        default=3000,
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
        default=1.0,
        help="Minimum nucleus top-p for sampling new solutions (default: 1.0).",
    )
    return parser


def main() -> None:
    """Performs offline self-improvement optimization."""

    ###
    # Initialize experiment
    ###
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed, deterministic=True)
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Store solutions as tuples of (SMILES, predicted_score) where predicted_score is a float used for ranking
    solutions: list[str] = []
    scores: list[float] = []
    threshold = get_optimization_threshold(args.target)

    ###
    # Initialize datasets, tokenizer and model
    ###
    datasets = load_docking_datasets(
        offline_dataset_config_path=args.offline_dataset_config,
        pretrain_dataset_config_path=args.pretrain_dataset_config,
        test_dataset_config_path=args.test_dataset_config,
        config_dump_dir=args.output_dir,
    )
    args.oracle_budget = args.oracle_budget - len(datasets["offline"])
    logger.info(
        f"Loaded {len(datasets['offline'])} solutions from offline dataset. Oracle budget updated to: {len(solutions)}"
    )

    tokenizer = load_tokenizer(args.tokenizer_config, config_dump_dir=args.output_dir)

    model = load_model(
        args.model_config,
        args.model_ckpt,
        args.device,
        config_dump_dir=args.output_dir,
    )

    ###
    # Self-improvement optimization rounds:
    ###
    for optimization_round in tqdm(
        range(args.num_self_improvement_rounds),
        desc="Self-improvement optimization rounds",
    ):

        ###
        # 1. Jointly update the model using the offline dataset
        ###
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
                dataset=(
                    datasets["pretrain"]
                    if task_name == "generation" and datasets["pretrain"] is not None
                    else datasets["offline"]
                ),
                tokenizer=tokenizer,
                task_name=task_name,
                batch_size=trainer.config.batch_size,
                num_workers=args.num_workers,
                device=args.device,
                shuffle=True,
            )
            val_loaders[task_name] = create_dataloader(
                dataset=datasets["test"],
                tokenizer=tokenizer,
                task_name=task_name,
                batch_size=trainer.config.batch_size,
                num_workers=args.num_workers,
                device=args.device,
                shuffle=False,
            )
        trainer.train(train_loaders=train_loaders, val_loaders=val_loaders)

        outputs = trainer.test(dataloader=val_loaders["prediction"])
        for target_transform in datasets["test"].target_transforms:
            y_true = target_transform.inverse_transform(
                outputs["true"].detach().cpu().numpy()
            )
            y_pred = target_transform.inverse_transform(
                outputs["predicted"].detach().cpu().numpy()
            )
        regression_metrics = calculate_regression_metrics(
            y_pred,
            y_true,
            out_dir=args.output_dir,
            optimization_round=optimization_round,
        )
        logger.info(
            f"Self-improvement round {optimization_round} regression metrics: {regression_metrics}"
        )
        free_memory(trainer)

        ###
        # 2. Sample new solutions from the model
        ###
        
        model.eval()
        _sampled_solutions, _sampled_scores = sample_solutions(
            tokenizer=tokenizer,
            model=model,
            advantage_fn=lambda x: x,
            oracle_fn=oracle_fn,
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
            target_transforms=datasets["offline"].target_transforms,
            rng=rng,
        )

        # Extend solutions with tuples of (SMILES, score)
        for solution, score in zip(_sampled_solutions, _sampled_scores):
            if solution not in solutions:
                solutions.append(solution)
                scores.append(score)
        
        ###
        # 3. Augment the offline dataset with new solutions
        ###
        datasets["offline"].append(
            sequences=_sampled_solutions, targets=_sampled_scores
        )
        logger.info(f"Augmented offline dataset with {len(_sampled_solutions)} solutions")

    ###
    # Evaluate final solutions
    ###
    logger.info(f"Sampled {len(solutions)} solutions")
    if len(solutions) > args.oracle_budget:
        # Sort by predicted score (descending, higher is better) and take top solutions
        # Zip solutions and scores together, sort by score, then unzip
        solutions_with_scores = list(zip(solutions, scores))
        solutions_with_scores.sort(key=lambda x: x[1], reverse=True)
        solutions_with_scores = solutions_with_scores[: args.oracle_budget]
        solutions, scores = zip(*solutions_with_scores)
        solutions = list(solutions)
        scores = list(scores)
        assert len(solutions) == args.oracle_budget

    objective_function_values = oracle_call(
        solutions, target=args.target, device=args.device
    )
    optimization_metrics = calculate_optimization_metrics(
        solutions, objective_function_values, threshold, out_dir=args.output_dir
    )

    logger.info(
        f"Optimization metrics for offline optimization: {optimization_metrics}"
    )
    save_solutions(solutions, objective_function_values, out_dir=args.output_dir)


if __name__ == "__main__":
    os.environ["QUICKVINA2_GPU_BINARY"] = (
        "/lustre/groups/aih/adam.izdebski/Vina-GPU-2.1/QuickVina2-GPU-2.1/QuickVina2-GPU-2-1"
    )
    if os.path.exists("/dev/shm"):
        os.environ["TMPDIR"] = "/dev/shm"
        print("Using /dev/shm (RAM disk) for temporary files to reduce Lustre I/O")
    else:
        os.environ["TMPDIR"] = "/tmp"
        print("Using /tmp for temporary files to reduce Lustre I/O")
    main()
