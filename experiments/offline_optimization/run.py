from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

if __package__ is None or __package__ == "":
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    _SRC_ROOT = _PROJECT_ROOT / "src"
    if _SRC_ROOT.exists() and str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))

# TODO: HFBackboneConfig and load_hf_backbone are not implemented in hyformer module
# from joint_improvement.hyformer import HFBackboneConfig, load_hf_backbone
from joint_improvement.utils.dataset_io import load_npz_dataset

from experiments.offline_optimization.conditional_sampling import (
    ConditionalSamplingConfig,
    run_conditional_sampling,
)
from experiments.offline_optimization.self_improvement import (
    SelfImprovementConfig,
    run_self_improvement,
)

LOGGER = logging.getLogger("offline_optimization")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Offline molecular optimisation pipeline with conditional sampling and self-improvement."),
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="NPZ file containing 'sequences' and optional 'scores'.",
    )
    parser.add_argument(
        "--oracle-path",
        type=Path,
        help=("Optional NPZ file used only for scoring generated samples (defaults to --data-path)."),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/offline_optimization"),
        help="Base directory for experiment outputs.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Overrides the default experiment folder name.",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "conditional_sampling", "self_improvement"],
        default="all",
        help="Select which stage(s) to execute.",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Explicit list of seeds to run.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of seeds to generate when --seeds is not provided.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Offsets auto-generated seeds.",
    )
    parser.add_argument(
        "--seed-index",
        type=int,
        help=("When launching via SBATCH arrays, provide the SLURM_ARRAY_TASK_ID and only that seed will run."),
    )

    # Backbone configuration
    parser.add_argument(
        "--backbone-model-id",
        type=str,
        default="mlp-classifier",
    )
    parser.add_argument("--backbone-task", type=str, default="causal-lm")
    parser.add_argument("--backbone-tokenizer-id", type=str)
    parser.add_argument("--backbone-revision", type=str)
    parser.add_argument("--backbone-cache-dir", type=str)
    parser.add_argument("--backbone-auth-token", type=str)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip backbone loading and emit deterministic placeholders.",
    )

    # Conditional sampling options
    parser.add_argument("--conditional-num-prompts", type=int, default=8)
    parser.add_argument("--conditional-samples-per-prompt", type=int, default=4)
    parser.add_argument(
        "--conditional-threshold",
        type=float,
        help="Keep molecules whose score exceeds this threshold.",
    )
    parser.add_argument(
        "--conditional-top-k",
        type=int,
        help="Restrict conditioning pool to top-k ranked by dataset scores.",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=("Given the molecule {sequence} with property score {score}, propose an improved molecule:"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")

    # Self-improvement options
    parser.add_argument("--self-top-k", type=int, default=32)
    parser.add_argument(
        "--self-min-score",
        type=float,
        help="Discard generated molecules below this score.",
    )

    return parser


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    dataset = load_npz_dataset(args.data_path)
    oracle_dataset = load_npz_dataset(args.oracle_path) if args.oracle_path else dataset

    seeds = resolve_seeds(args)
    default_name = args.experiment_name or args.data_path.stem
    experiment_dir = prepare_experiment_dir(args.output_dir, default_name)

    backbone_bundle = None
    if args.stage in ("all", "conditional_sampling") and not args.dry_run:
        backbone_bundle = load_backbone_from_args(args)

    experiment_summary: list[dict[str, Any]] = []
    all_conditional_rows: list[dict[str, Any]] = []

    for seed in seeds:
        LOGGER.info("==== Seed %s ====", seed)
        seed_dir = experiment_dir / f"seed_{seed:04d}"
        conditional_dir = seed_dir / "conditional_sampling"
        self_dir = seed_dir / "self_improvement"

        conditional_rows: list[dict[str, Any]] | None = None
        conditional_summary: dict[str, Any] | None = None
        self_summary: dict[str, Any] | None = None

        if args.stage in ("all", "conditional_sampling"):
            if backbone_bundle is None and not args.dry_run:
                raise RuntimeError(
                    "Backbone failed to load but conditional sampling was requested.",
                )
            cond_cfg = ConditionalSamplingConfig(
                seed=seed,
                sequences=dataset.sequences,
                scores=dataset.scores,
                backbone=backbone_bundle,
                output_csv=conditional_dir / "candidates.csv",
                output_json=conditional_dir / "summary.json",
                prompt_template=args.prompt_template,
                num_prompts=args.conditional_num_prompts,
                samples_per_prompt=args.conditional_samples_per_prompt,
                property_threshold=args.conditional_threshold,
                top_k=args.conditional_top_k,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                device=args.device,
                dry_run=args.dry_run,
                model_id=args.backbone_model_id,
                score_lookup=oracle_dataset.score_lookup,
                extra_metadata={"data_path": str(args.data_path)},
            )
            cond_result = run_conditional_sampling(cond_cfg)
            conditional_rows = cond_result.rows
            conditional_summary = read_json(cond_result.json_path)
            all_conditional_rows.extend(conditional_rows)
        else:
            conditional_csv = conditional_dir / "candidates.csv"
            conditional_rows = load_rows_from_csv(conditional_csv)
            if conditional_rows:
                conditional_summary_path = conditional_dir / "summary.json"
                conditional_summary = read_json(conditional_summary_path)

        if args.stage in ("all", "self_improvement"):
            if not conditional_rows:
                LOGGER.warning(
                    "No conditional candidates for seed %s; skipping self-improvement.",
                    seed,
                )
            else:
                self_cfg = SelfImprovementConfig(
                    seed=seed,
                    top_k=args.self_top_k,
                    output_csv=self_dir / "top_candidates.csv",
                    output_json=self_dir / "summary.json",
                    augmented_npz=self_dir / "augmented_dataset.npz",
                    min_score=args.self_min_score,
                    extra_metadata={"data_path": str(args.data_path)},
                )
                self_result = run_self_improvement(self_cfg, conditional_rows)
                self_summary = read_json(self_result.json_path)

        experiment_summary.append(
            {
                "seed": seed,
                "conditional_sampling": conditional_summary,
                "self_improvement": self_summary,
            }
        )

    write_master_summary(experiment_dir, args, experiment_summary)

    if all_conditional_rows:
        aggregate_path = experiment_dir / "conditional_candidates.csv"
        write_rows_to_csv(aggregate_path, all_conditional_rows)
        LOGGER.info("Wrote aggregated conditional candidates to %s", aggregate_path)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def load_backbone_from_args(args: argparse.Namespace):
    # TODO: Implement HFBackboneConfig and load_hf_backbone in hyformer module
    # For now, this function is not implemented
    raise NotImplementedError(
        "HFBackboneConfig and load_hf_backbone are not yet implemented. "
        "Use Hyformer and HyformerConfig directly instead."
    )
    # config = HFBackboneConfig(
    #     model_id=args.backbone_model_id,
    #     task=args.backbone_task,
    #     tokenizer_id=args.backbone_tokenizer_id,
    #     revision=args.backbone_revision,
    #     cache_dir=args.backbone_cache_dir,
    #     trust_remote_code=args.trust_remote_code,
    #     use_auth_token=args.backbone_auth_token,
    # )
    # bundle = load_hf_backbone(config)
    # LOGGER.info("Loaded backbone %s", args.backbone_model_id)
    # return bundle


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        seeds = list(dict.fromkeys(args.seeds))
    else:
        seeds = [args.seed_offset + idx for idx in range(args.num_seeds)]
    if args.seed_index is not None:
        if args.seed_index < 0 or args.seed_index >= len(seeds):
            raise ValueError(
                f"seed_index {args.seed_index} is invalid for {len(seeds)} seeds.",
            )
        seeds = [seeds[args.seed_index]]
    return seeds


def prepare_experiment_dir(output_dir: Path, experiment_name: str | None) -> Path:
    if experiment_name:
        exp_dir = output_dir / experiment_name
    else:
        exp_dir = output_dir / "default"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def load_rows_from_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        LOGGER.warning("Conditional sampling file not found: %s", path)
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        data = csv.DictReader(handle)
        for raw_row in data:
            row = dict(raw_row)
            for key in ("seed", "prompt_index", "sample_index"):
                if key in row and row[key] != "":
                    row[key] = int(row[key])
            for key in ("conditioning_score", "score"):
                if key in row:
                    row[key] = _safe_float(row[key])
            rows.append(row)
    return rows


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_master_summary(
    experiment_dir: Path,
    args: argparse.Namespace,
    per_seed: list[dict[str, Any]],
) -> None:
    summary_path = experiment_dir / "summary.json"
    payload = {
        "data_path": str(args.data_path),
        "oracle_path": str(args.oracle_path) if args.oracle_path else None,
        "stage": args.stage,
        "seeds": [entry["seed"] for entry in per_seed],
        "backbone_model_id": args.backbone_model_id,
        "conditional": {
            "num_prompts": args.conditional_num_prompts,
            "samples_per_prompt": args.conditional_samples_per_prompt,
            "threshold": args.conditional_threshold,
            "top_k": args.conditional_top_k,
        },
        "self_improvement": {
            "top_k": args.self_top_k,
            "min_score": args.self_min_score,
        },
        "seeds_summary": per_seed,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    LOGGER.info("Wrote experiment summary to %s", summary_path)


def write_rows_to_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    fieldnames = [
        "seed",
        "model_id",
        "prompt_index",
        "sample_index",
        "conditioning_sequence",
        "conditioning_score",
        "prompt",
        "generated_sequence",
        "score",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            if csv_row.get("conditioning_score") is None:
                csv_row["conditioning_score"] = ""
            if csv_row.get("score") is None:
                csv_row["score"] = ""
            writer.writerow(csv_row)


if __name__ == "__main__":
    main()
