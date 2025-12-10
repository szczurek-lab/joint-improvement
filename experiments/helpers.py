"""Helper functions for experiment scripts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import argparse

from loguru import logger

from joint_improvement.hyformer import Hyformer, HyformerConfig
from joint_improvement.tokenizers import SMILESTokenizer, SMILESTokenizerConfig
from joint_improvement.trainers.multitask import MultiTaskTrainer, TrainerConfig
from joint_improvement.utils import (
    LMCollator,
    MLMCollator,
    PredictionCollator,
    SequenceDataLoader,
    SequenceDataset,
    SequenceDatasetConfig,
)


def load_model(
    model_config: Path,
    model_ckpt: Path | None = None,
    device: str = "cpu",
    config_dump_dir: Path | None = None,
) -> Hyformer:
    """Load or initialize Hyformer model.

    Parameters
    ----------
    model_config : Path
        Path to model configuration JSON file.
    model_ckpt : Path, optional
        Path to model checkpoint to load weights from.
    device : str, default="cpu"
        Device to load model on.
    config_dump_dir : Path, optional
        Directory to save configuration for reproducibility. If provided,
        saves the loaded model configuration as model_config.json in this directory.

    Returns
    -------
    Hyformer
        Initialized or loaded model.
    """
    config = HyformerConfig.from_pretrained(model_config)

    if config_dump_dir is not None:
        dump_configs(config_dump_dir, {"model_config": config})

    model = Hyformer.from_config(config)

    if model_ckpt is not None:
        if not model_ckpt.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_ckpt}")
        model.load_pretrained(model_ckpt, device=device)
        logger.info("Loaded model weights from checkpoint")
    else:
        logger.info("Initialized new model from scratch")

    logger.info(f"Model parameters: {model.get_num_params(trainable_only=True):,}")

    return model


def load_tokenizer(tokenizer_config: Path, config_dump_dir: Path | None = None) -> SMILESTokenizer:
    """Load SMILES tokenizer.

    Parameters
    ----------
    tokenizer_config : Path
        Path to tokenizer directory containing tokenizer_config.json and vocab file.
    config_dump_dir : Path, optional
        Directory to save configuration for reproducibility. If provided,
        saves the loaded tokenizer configuration as tokenizer_config.json in this directory.

    Returns
    -------
    SMILESTokenizer
        Loaded tokenizer.
    """
    config = SMILESTokenizerConfig.from_pretrained(tokenizer_config)

    if config_dump_dir is not None:
        dump_configs(config_dump_dir, {"tokenizer_config": config})

    tokenizer = SMILESTokenizer.from_pretrained(tokenizer_config)
    return tokenizer


def load_dataset(dataset_config_path: Path, config_dump_dir: Path | None = None) -> SequenceDataset:
    """Load a single dataset from configuration file.

    Parameters
    ----------
    dataset_config_path : Path
        Path to SequenceDatasetConfig JSON file.
    config_dump_dir : Path, optional
        Directory to save configuration for reproducibility. If provided,
        saves the loaded dataset configuration as dataset_config.json in this directory.

    Returns
    -------
    SequenceDataset
        Loaded dataset.
    """
    config = SequenceDatasetConfig.from_pretrained(dataset_config_path)

    if config_dump_dir is not None:
        dump_configs(config_dump_dir, {"dataset_config": config})

    dataset = SequenceDataset.from_config(config)
    return dataset


def create_dataloader(
    dataset: SequenceDataset,
    tokenizer: SMILESTokenizer,
    task_name: str,
    batch_size: int,
    num_workers: int,
    device: str,
    shuffle: bool,
) -> SequenceDataLoader:
    """Create a single dataloader from dataset and tokenizer.

    Parameters
    ----------
    dataset : SequenceDataset
        Dataset instance.
    collator : LMCollator | MLMCollator
        Collator instance for batching.
    batch_size : int
        Batch size.
    num_workers : int
        Number of data loading workers.
    shuffle : bool
        Whether to shuffle the dataset.
    drop_last : bool
        Whether to drop the last incomplete batch.

    Returns
    -------
    SequenceDataLoader
        Configured dataloader.
    """
    if task_name == "lm":
        collator = LMCollator(tokenizer=tokenizer)
    elif task_name == "mlm":
        collator = MLMCollator(tokenizer=tokenizer)
    elif task_name == "prediction":
        collator = PredictionCollator(tokenizer=tokenizer)
    else:
        raise ValueError(f"Unknown task: {task_name}. Must be 'lm', 'mlm' or 'prediction'.")

    return SequenceDataLoader(
        dataset=dataset,
        collator=collator,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def load_trainer(
    trainer_config: Path,
    model: Hyformer,
    device: str,
    out_dir: Path,
    config_dump_dir: Path | None = None,
) -> MultiTaskTrainer:
    """Load or initialize MultiTaskTrainer.

    Parameters
    ----------
    trainer_config : Path
        Path to TrainerConfig JSON file.
    model : Hyformer
        Model to train.
    device : str
        Device to train on.
    out_dir : Path
        Output directory to save checkpoints and logs.
    config_dump_dir : Path, optional
        Directory to save configuration for reproducibility. If provided,
        saves the loaded trainer configuration as trainer_config.json in this directory.

    Returns
    -------
    MultiTaskTrainer
        Initialized trainer instance.
    """
    config = TrainerConfig.from_pretrained(trainer_config)

    if config_dump_dir is not None:
        dump_configs(config_dump_dir, {"trainer_config": config})

    trainer = MultiTaskTrainer(
        config=config,
        model=model,
        device=device,
        out_dir=out_dir,
    )

    return trainer


def dump_configs(
    output_dir: Path,
    configs: dict[str, Any],
) -> None:
    """Save all configurations to JSON files in output directory.

    Parameters
    ----------
    output_dir : Path
        Output directory to save configs to.
    configs : dict[str, Any]
        Dictionary mapping config names to config objects. Config objects should
        have a `to_json()` method or be dataclasses that can be converted with `asdict()`.
        Each key will be saved as `{key}.json` in the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for config_name, config_obj in configs.items():
        config_path = output_dir / f"{config_name}.json"

        # Try to_json() method first (for config classes)
        if hasattr(config_obj, "to_json"):
            config_obj.to_json(config_path)
            logger.info(f"Saved {config_name}: {config_path}")
        # Fall back to dataclass asdict
        elif hasattr(config_obj, "__dataclass_fields__"):
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(asdict(config_obj), f, indent=2, default=str)
        else:
            raise ValueError(
                f"Config '{config_name}' has no to_json() method or dataclass fields. Cannot serialize to JSON."
            )


def dump_args(
    args: argparse.Namespace,
    output_dir: Path,
    filename: str = "command_line_args.json",
) -> None:
    """Save command-line arguments to a JSON file for reproducibility.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments namespace from argparse.
    output_dir : Path
        Output directory to save arguments to.
    filename : str, default="command_line_args.json"
        Name of the JSON file to save arguments to.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    args_dict = vars(args)
    args_path = output_dir / filename

    with args_path.open("w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2, default=str)

    logger.info(f"Saved command-line arguments: {args_path}")
