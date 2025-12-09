from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

import numpy as np

_LOGGER = logging.getLogger(__name__)


@dataclass
class SelfImprovementConfig:
    """Configuration for the self-improvement stage."""

    seed: int
    top_k: int
    output_csv: Path
    output_json: Path
    augmented_npz: Path | None = None
    min_score: float | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelfImprovementResult:
    """Outputs created by the self-improvement stage."""

    rows: list[dict[str, Any]]
    csv_path: Path
    json_path: Path
    augmented_npz: Path | None


def run_self_improvement(
    config: SelfImprovementConfig,
    generated_rows: Iterable[dict[str, Any]],
) -> SelfImprovementResult:
    """Select the best candidates and optionally build an augmented dataset."""
    _LOGGER.info("Running self-improvement | seed=%s", config.seed)
    filtered_rows = _filter_scored_rows(generated_rows, config.min_score)
    if config.top_k > 0:
        filtered_rows = filtered_rows[: config.top_k]

    _write_rows_to_csv(config.output_csv, filtered_rows)
    summary = _summarise(filtered_rows, config)
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    with config.output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    augmented_path = None
    if filtered_rows and config.augmented_npz is not None:
        augmented_path = _write_augmented_npz(config.augmented_npz, filtered_rows)

    _LOGGER.info(
        "Self-improvement finished | seed=%s kept=%s best_score=%s",
        config.seed,
        len(filtered_rows),
        summary["best_score"],
    )

    return SelfImprovementResult(
        rows=filtered_rows,
        csv_path=config.output_csv,
        json_path=config.output_json,
        augmented_npz=augmented_path,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter_scored_rows(
    rows: Iterable[dict[str, Any]],
    min_score: float | None,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        score = row.get("score")
        if isinstance(score, (int, float)):
            score_value = float(score)
        else:
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                continue

        if min_score is not None and score_value < min_score:
            continue

        enriched = dict(row)
        enriched["score"] = score_value
        scored.append(enriched)

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


def _write_rows_to_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
            csv_row = row.copy()
            if csv_row.get("conditioning_score") is None:
                csv_row["conditioning_score"] = ""
            writer.writerow(csv_row)


def _summarise(
    rows: list[dict[str, Any]],
    config: SelfImprovementConfig,
) -> dict[str, Any]:
    if rows:
        scores = [row["score"] for row in rows]
        best_row = rows[0]
        best_payload = {
            "generated_sequence": best_row["generated_sequence"],
            "score": best_row["score"],
            "prompt_index": best_row["prompt_index"],
            "sample_index": best_row["sample_index"],
        }
    else:
        scores = []
        best_payload = None

    summary: dict[str, Any] = {
        "seed": config.seed,
        "top_k": config.top_k,
        "kept": len(rows),
        "best_score": float(scores[0]) if scores else None,
        "mean_score": float(np.mean(scores)) if scores else None,
        "median_score": float(np.median(scores)) if scores else None,
        "unique_sequences": len({row["generated_sequence"] for row in rows}),
        "best_candidate": best_payload,
        "metadata": config.extra_metadata,
    }
    return summary


def _write_augmented_npz(path: Path, rows: list[dict[str, Any]]) -> Path:
    sequences = np.array(
        [row["generated_sequence"] for row in rows],
        dtype=object,
    )
    scores = np.array(
        [row["score"] for row in rows],
        dtype=np.float32,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, sequences=sequences, scores=scores)
    return path


__all__ = [
    "SelfImprovementConfig",
    "SelfImprovementResult",
    "run_self_improvement",
]
