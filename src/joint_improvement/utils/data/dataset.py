from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass
class SequenceDatasetConfig:
    """Configuration for creating a SequenceDataset.

    Parameters
    ----------
    data_path : Path | str, optional
        Data path or template path to the NPZ archive file.
        Use `{split}` and `{seed}` placeholders to override
         split and seed respectively. Example: `"seed_{seed}/{split}.npz"`.
    sequence_key : str, optional
        Key name for the sequence data in the NPZ archive.
    target_key : str, optional
        Key name for the target data in the NPZ archive.
    transforms : Sequence[dict[str, Any]], optional
        Sequence of transform configuration dictionaries.
    target_transforms : Sequence[dict[str, Any]], optional
        Sequence of target transform configuration dictionaries.

    """

    data_path: Path | str | None = None
    sequence_key: str | None = None
    target_key: str | None = None
    transforms: Sequence[dict[str, Any]] | None = None
    target_transforms: Sequence[dict[str, Any]] | None = None

    @classmethod
    def from_pretrained(cls, pretrained_config_name_or_path: Path | str) -> SequenceDatasetConfig:
        """Load SequenceDatasetConfig from a JSON file.

        Parameters
        ----------
        pretrained_config_name_or_path : Path | str
            Path to the JSON configuration file.

        Returns
        -------
        SequenceDatasetConfig
            Configuration object loaded from JSON.

        Raises
        ------
        FileNotFoundError
            If the JSON file does not exist.
        json.JSONDecodeError
            If the JSON file is invalid.

        Examples
        --------
        >>> config = SequenceDatasetConfig.from_pretrained("configs/datasets/zinc250k/logp/seed_0/dataset_config.json")
        >>> dataset = SequenceDataset.from_config(config, split="train")
        """
        config_path = Path(pretrained_config_name_or_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(**data)


class SequenceDataset(Dataset):
    """Minimal PyTorch Dataset.

    Follows PyTorch conventions by exposing `transforms` and `target_transforms`
    applied on-the-fly during `__getitem__`.
    """

    SEQUENCE_FIELD: str = "sequence"
    TARGET_FIELD: str = "labels"

    def __init__(
        self,
        sequences: Sequence[str],
        targets: Sequence[float | int],
        transforms: Sequence[Callable[[str], str]] | None = None,
        target_transforms: Sequence[Callable[[float | int], float | int]] | None = None,
    ) -> None:
        """Initialize SequenceDataset.

        Parameters
        ----------
        sequences : Sequence[str]
            Input sequences. Must have the same length as `targets`.
        targets : Sequence[float | int]
            Target values (labels for classification, continuous values for regression).
            Must have the same length as `sequences`.
        transforms : Sequence[Callable[[str], str]], optional
            Optional sequence of transforms applied to each sequence during `__getitem__`.
            Transforms are applied in order and must return strings (e.g., for data augmentation).
            Tokenization is handled separately in collators, not in transforms.
        target_transforms : Sequence[Callable[[float | int], float | int]], optional
            Optional sequence of transforms applied to each target during `__getitem__`.
            Transforms are applied in order.

        Raises
        ------
        ValueError
            If `sequences` and `targets` have different lengths.

        Examples
        --------
        >>> dataset = SequenceDataset(sequences=["ACDEFGHIK", "LMNPQRSTV"], targets=[0.5, 0.8])
        """
        if len(sequences) != len(targets):
            raise ValueError("`sequences` and `targets` must have the same length.")

        self._sequences = list(sequences)
        self._targets = list(targets)
        self.transforms = transforms
        self.target_transforms = target_transforms

    @property
    def sequences(self) -> list[str]:
        """Return the sequences.

        Returns
        -------
        list[str]
            Sequences.
        """
        return self._sequences

    @property
    def targets(self) -> list[float | int]:
        """Return the targets.

        Returns
        -------
        list[float | int]
            Targets.
        """
        return self._targets

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self._sequences)

    def __getitem__(self, index: int) -> dict[str, float | int | str]:
        """Get a single sample from the dataset.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve (0-based).

        Returns
        -------
        dict[str, float | int | str]
            Dictionary containing:
            - `SEQUENCE_FIELD`: The sequence (possibly transformed)
            - `TARGET_FIELD`: The target (possibly transformed)

        Examples
        --------
        >>> dataset = SequenceDataset(sequences=["ACDEFGHIK"], targets=[0.5])
        >>> sample = dataset[0]
        >>> sample["sequence"]
        'ACDEFGHIK'
        >>> sample["target"]
        0.5
        """
        sequence = self._sequences[index]
        target: float | int = self._targets[index]

        if self.transforms is not None:
            for transform in self.transforms:
                sequence = transform(sequence)

        if self.target_transforms is not None:
            for target_transform in self.target_transforms:
                target = target_transform(target)  # type: ignore[assignment, arg-type]

        return {
            self.SEQUENCE_FIELD: sequence,
            self.TARGET_FIELD: target,
        }

    @classmethod
    def from_config(
        cls,
        config: SequenceDatasetConfig,
        root: Path | str | None = None,
        split: Literal["train", "val", "test"] | None = None,
    ) -> SequenceDataset:
        """Create a SequenceDataset from a configuration.

        Parameters
        ----------
        config : SequenceDatasetConfig
            Configuration for the SequenceDataset.
        root : Path | str, optional
            Root directory for data files. If provided, paths will be resolved
            relative to this root.
        split : Literal["train", "val", "test"], optional
            Dataset split name. Required if `data_path` contains `{split}` placeholder.
            Replaces `{split}` with the split name.

        Returns
        -------
        SequenceDataset

        """
        from .data_io import load_npz_data
        from .target_transforms import create_target_transform

        if config.data_path is None:
            raise ValueError("data_path must be provided in config")

        data_path = _resolve_data_path(
            data_path=config.data_path,
            split=split,
            root=root,
        )

        # Load data (load_npz_data will raise FileNotFoundError if path doesn't exist)
        data = load_npz_data(
            data_path,
            sequence_key=config.sequence_key,
            target_key=config.target_key,
        )

        sequences = data.get(config.sequence_key) if config.sequence_key else None
        targets = data.get(config.target_key) if config.target_key else None

        if config.sequence_key and sequences is None:
            raise ValueError(f"Sequence key '{config.sequence_key}' not found in loaded data")
        if config.target_key and targets is None:
            raise ValueError(f"Target key '{config.target_key}' not found in loaded data")

        if sequences is None:
            raise ValueError("sequence_key must be provided in config")

        transforms: list[Callable[[str], str]] | None = None
        if config.transforms:
            # TODO: Implement create_transform when transforms module is populated
            transforms = []

        target_transforms = None
        if config.target_transforms:
            target_transforms = [
                create_target_transform(transform_config) for transform_config in config.target_transforms
            ]

        return cls(
            sequences=sequences,
            targets=targets or [],
            transforms=transforms,
            target_transforms=target_transforms,
        )


def _resolve_data_path(
    data_path: Path | str,
    split: Literal["train", "val", "test"] | None = None,
    root: Path | str | None = None,
) -> Path:
    """Resolve data path by replacing placeholders with actual values.

    Parameters
    ----------
    data_path : Path | str
        Path template containing placeholders. Use `{split}` as a placeholder for split
        name. Example: `"directory/seed_0/{split}.npz"`.
    split : Literal["train", "val", "test"], optional
        Dataset split name. Required if `data_path` contains `{split}` placeholder.
        Replaces `{split}` with the split name.
    root : Path | str, optional
        Root directory for data files. If provided, paths will be resolved
        relative to this root.

    Returns
    -------
    Path
        Resolved data path with all placeholders replaced.

    Raises
    ------
    ValueError
        If required placeholders are missing when corresponding parameters are provided.
    """
    data_path_str = str(data_path)

    if split is not None:
        placeholder = "{split}"
        if placeholder not in data_path_str:
            raise ValueError(f"data_path must contain {placeholder} as a placeholder when split is provided. ")
        data_path_str = data_path_str.replace(placeholder, split)

    data_path = Path(data_path_str)

    if root is not None:
        root_path = Path(root)
        data_path = root_path / data_path

    return data_path
