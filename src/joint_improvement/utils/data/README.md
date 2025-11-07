# Data Utilities

This module provides utilities for loading and processing data in NPZ (NumPy archive) format.


## Dataset Usage

```python
from joint_improvement.utils import SequenceDataset, SequenceDatasetConfig

config = SequenceDatasetConfig(
    data_path="seed_{seed}/{split}.npz",  # Any path template with {seed} and/or {split} placeholders
    sequence_key="sequences",
    target_key="targets"
)

# Placeholders are replaced automatically when parameters are provided
train_dataset = SequenceDataset.from_config(config, split="train", seed=42)  # seed_42/train.npz
val_dataset = SequenceDataset.from_config(config, split="val", seed=42, root="/data")  # /data/seed_42/val.npz

# Alternatively, no placeholders
config = SequenceDatasetConfig(
    data_path="data/train.npz",  # Direct path without placeholders
    sequence_key="sequences",
    target_key="targets"
)
dataset = SequenceDataset.from_config(config)  # Loads data/train.npz
```

Placeholders can appear anywhere in the path. Examples:
- `"seed_{seed}/{split}.npz"` → `seed_42/train.npz`
- `"data/{seed}/splits/{split}.npz"` → `data/42/splits/train.npz`
- `"experiment_{seed}_{split}.npz"` → `experiment_42_train.npz`

## NPZ File Format

NPZ files contain optional `sequences` and `targets` arrays, plus any additional fields. Specify keys via `sequence_key` and `target_key` in config.

## Module Components

- `dataset.py`: `SequenceDataset` class
- `data_io.py`: NPZ file I/O
- `target_transforms.py`: Target transform utilities
- `transforms.py`: Sequence transform utilities
- `collator.py`: Batching utilities
- `dataloader.py`: DataLoader utilities
