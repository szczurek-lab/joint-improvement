"""Build docking dataset."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np

# Ensure experiments package is importable
if (project_root := Path(__file__).parent.parent.parent) not in [Path(p) for p in sys.path]:
    sys.path.insert(0, str(project_root))

from joint_improvement.utils import set_seed  # noqa: E402
from joint_improvement.utils.chemistry import (  # noqa: E402
    calculate_docking_batch,
    calculate_qed_batch,
    calculate_sa_batch,
)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Build docking dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save dataset.",
    )
    parser.add_argument(
        "--train-dataset-path",
        type=Path,
        required=True,
        help="Path to input dataset.",
    )
    parser.add_argument(
        "--test-dataset-path",
        type=Path,
        required=True,
        help="Path to test dataset.",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target to optimize for.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for reproducibility (default: 0).",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=1500,
        help="Size of the dataset to build (default: 1500).",
    )
    return parser


def main() -> None:
    """Main training function."""
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed, deterministic=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_sequences = np.load(args.train_dataset_path)["sequence"].tolist()
    assert len(train_sequences) >= args.dataset_size
    train_sequences = random.sample(train_sequences, args.dataset_size)

    test_sequences = np.load(args.test_dataset_path)["sequence"].tolist()

    for split in ["train", "test"]:
        sequences = train_sequences if split == "train" else test_sequences

        # Compute properties
        qed = np.asarray(calculate_qed_batch(sequences), dtype=np.float32)
        sa = np.asarray(calculate_sa_batch(sequences), dtype=np.float32)
        docking = np.asarray(
            calculate_docking_batch(sequences, target=args.target, device="cpu"),
            dtype=np.float32,
        )

        # Check for invalid values
        invalid_mask = (~np.isfinite(docking)) | (~np.isfinite(sa)) | (~np.isfinite(qed))
        assert not invalid_mask.any(), "Found invalid values in docking, sa, or qed"

        # Concatenate targets: [docking, sa, qed]
        target = np.concatenate(
            [docking.reshape(-1, 1), sa.reshape(-1, 1), qed.reshape(-1, 1)], axis=1
        )

        # Save dataset
        output_path = os.path.join(args.output_dir, f"{split}.npz")
        np.savez(
            output_path,
            sequence=np.asarray(sequences, dtype=str),
            target=np.asarray(target, dtype=np.float32),
        )
        print(f"Saved {split} dataset: {output_path} ({len(sequences)} sequences)")

if __name__ == "__main__":
    main()
