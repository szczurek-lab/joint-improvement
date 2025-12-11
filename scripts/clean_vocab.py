#!/usr/bin/env python3
"""Utility script to clean vocabulary files by removing special tokens.

Special tokens should be defined in tokenizer_config.json, not in vocab files.
This script removes special tokens from vocab files to ensure consistent behavior.

Usage:
    python scripts/clean_vocab.py <vocab_file> [--output <output_file>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Default special tokens that should not be in vocab files
DEFAULT_SPECIAL_TOKENS = {
    "<bos>",
    "<eos>",
    "<pad>",
    "<unk>",
    "<mask>",
}


def clean_vocab_file(
    vocab_file: Path,
    output_file: Path | None = None,
    special_tokens: set[str] | None = None,
) -> None:
    """Remove special tokens from vocabulary file.

    Parameters
    ----------
    vocab_file : Path
        Path to input vocabulary file.
    output_file : Path | None, optional
        Path to output file. If None, overwrites input file.
    special_tokens : set[str] | None, optional
        Set of special tokens to remove. If None, uses DEFAULT_SPECIAL_TOKENS.
    """
    if special_tokens is None:
        special_tokens = DEFAULT_SPECIAL_TOKENS

    if not vocab_file.exists():
        print(f"Error: Vocabulary file not found: {vocab_file}", file=sys.stderr)
        sys.exit(1)

    # Read vocab file and filter out special tokens
    cleaned_tokens: list[str] = []
    removed_tokens: list[str] = []

    with open(vocab_file, encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if not token:
                continue

            if token in special_tokens:
                removed_tokens.append(token)
            else:
                cleaned_tokens.append(token)

    # Write cleaned vocab
    output_path = output_file or vocab_file
    with open(output_path, "w", encoding="utf-8") as f:
        for token in cleaned_tokens:
            f.write(f"{token}\n")

    print(f"Cleaned vocabulary file: {vocab_file}")
    print(f"  Removed {len(removed_tokens)} special tokens: {removed_tokens}")
    print(f"  Kept {len(cleaned_tokens)} regular tokens")
    print(f"  Output written to: {output_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Remove special tokens from vocabulary files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean vocab file in place
  python scripts/clean_vocab.py data/vocabularies/zinc250k.txt

  # Clean vocab file to new file
  python scripts/clean_vocab.py data/vocabularies/zinc250k.txt --output data/vocabularies/zinc250k_cleaned.txt
        """,
    )
    parser.add_argument(
        "vocab_file",
        type=Path,
        help="Path to vocabulary file to clean",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path (default: overwrite input file)",
    )
    parser.add_argument(
        "--special-tokens",
        nargs="+",
        default=None,
        help="Additional special tokens to remove (default: uses built-in list)",
    )

    args = parser.parse_args()

    special_tokens = DEFAULT_SPECIAL_TOKENS.copy()
    if args.special_tokens:
        special_tokens.update(args.special_tokens)

    clean_vocab_file(
        vocab_file=args.vocab_file,
        output_file=args.output,
        special_tokens=special_tokens,
    )


if __name__ == "__main__":
    main()
