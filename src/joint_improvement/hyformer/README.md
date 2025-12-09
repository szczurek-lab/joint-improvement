# Hyformer

A flexible transformer backbone model that supports both causal (autoregressive) and bidirectional attention modes, designed for multi-task learning in chemistry applications.

## Overview

Hyformer is a transformer architecture that can function as both an encoder and decoder through configurable attention modes. It supports multiple tasks including:

- **Language Modeling (LM)**: Causal/autoregressive next-token prediction
- **Masked Language Modeling (MLM)**: Bidirectional masked token prediction
- **Prediction Tasks**: Classification and regression for downstream tasks

The model uses weight tying between embeddings and language modeling heads to reduce parameters and improve performance, following LLaMA-style initialization.

## Features

- 🔄 **Hybrid Attention**: Seamlessly switch between causal (decoder) and bidirectional (encoder) attention
- 🎯 **Multi-Task Support**: Built-in support for LM, MLM, and prediction tasks
- 🧠 **Weight Tying**: Shared weights between embeddings and language modeling heads
- ⚡ **KV Caching**: Efficient autoregressive generation with key-value caching
- 🚀 **HuggingFace Compatible**: Save/load models using HuggingFace Hub format
- 📦 **Modular Design**: Clean separation of layers, losses, and utilities
- 🔧 **Configurable**: Easy configuration via JSON files or programmatic config

## Installation

Hyformer is part of the `joint-improvement` package. Install dependencies:

```bash
# Install from project root
pip install -e .

# Or install specific dependencies
pip install torch>=2.0.0 safetensors huggingface-hub
```

## Quick Start

### Basic Usage

```python
import torch
from joint_improvement.hyformer import Hyformer, HyformerConfig

# Create model from config
config = HyformerConfig(
    vocab_size=32000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    max_seq_len=128,
)

model = Hyformer.from_config(config)

# Forward pass for language modeling (causal)
input_ids = torch.randint(0, 32000, (2, 10))  # batch_size=2, seq_len=10
outputs = model(input_ids, task="lm")
print(outputs.logits.shape)  # [2, 10, 32000]

# Forward pass for masked language modeling (bidirectional)
attention_mask = torch.ones(2, 10)
outputs = model(input_ids, task="mlm", attention_mask=attention_mask)
print(outputs.logits.shape)  # [2, 10, 32000]
```

### Loading from Configuration File

```python
from joint_improvement.hyformer import Hyformer, HyformerConfig

# Load config from JSON
config = HyformerConfig.from_pretrained("configs/hyformer/base.json")
model = Hyformer.from_config(config)
```

### Loading Pretrained Models

```python
from joint_improvement.hyformer import Hyformer

# Load from HuggingFace Hub
model = Hyformer.from_pretrained(
    "username/hyformer-model",
    device="cuda",
)

# Load with custom config
from joint_improvement.hyformer import HyformerConfig

config = HyformerConfig(vocab_size=32000, d_model=512, n_heads=8, n_layers=6)
model = Hyformer.from_pretrained(
    "username/hyformer-model",
    config=config,
    strict=False,  # Allow missing/unexpected keys
)
```

### Training with Different Tasks

```python
import torch
from joint_improvement.hyformer import Hyformer, HyformerConfig

model = Hyformer.from_config(HyformerConfig(vocab_size=32000, d_model=512))

# Language Modeling (causal LM)
input_ids = torch.randint(0, 32000, (2, 10))
labels = input_ids.clone()  # For next-token prediction
outputs = model(input_ids, task="lm", labels=labels)
loss = outputs.loss  # Computed automatically

# Masked Language Modeling
labels = torch.randint(0, 32000, (2, 10))
labels[labels == 0] = -1  # Mask some tokens
outputs = model(input_ids, task="mlm", labels=labels, attention_mask=attention_mask)
loss = outputs.loss

# Prediction task (requires num_prediction_tasks in constructor)
model = Hyformer(
    vocab_size=32000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    max_seq_len=128,
    num_prediction_tasks=10
)
labels = torch.randint(0, 10, (2,))  # Classification labels
outputs = model(input_ids, task="prediction", labels=labels, attention_mask=attention_mask)
loss = outputs.loss
```

## Architecture

### Model Structure

```
Input Tokens
    ↓
Token Embeddings (weight-tied with heads)
    ↓
Transformer Blocks (x N)
    ├─ Self-Attention (causal or bidirectional)
    ├─ RMSNorm
    └─ SwiGLU MLP
    ↓
Final LayerNorm
    ↓
Task-Specific Heads
    ├─ "lm": Language modeling head
    ├─ "mlm": Masked language modeling head
    └─ "prediction": Optional prediction head
```

### Key Components

- **Transformer Blocks**: Pre-normalization architecture with RMSNorm and residual connections
- **Rotary Positional Embeddings (RoPE)**: Relative positional encoding
- **SwiGLU MLP**: Gated linear unit with Swish activation
- **Task Heads**: Separate heads for different tasks, with weight tying for LM/MLM

## API Reference

### HyformerConfig

Configuration dataclass for model parameters.

```python
@dataclass
class HyformerConfig:
    vocab_size: int           # Vocabulary size (required)
    d_model: int = 512        # Model dimension
    n_heads: int = 8          # Number of attention heads
    n_layers: int = 6         # Number of transformer layers
    max_seq_len: int = 128    # Maximum sequence length
    attn_dropout: float = 0.0 # Attention dropout
    resid_dropout: float = 0.0 # Residual dropout
    eps: float = 1e-6         # RMSNorm epsilon
```

**Methods:**
- `from_json(path)`: Load config from JSON file
- `from_pretrained(path)`: Load config from JSON (alias for `from_json`)
- `to_json(path)`: Save config to JSON file
- `to_dict()`: Convert to dictionary

### Hyformer

Main model class.

```python
class Hyformer(PretrainedMixin, nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int,
        num_prediction_tasks: int | None = None,
    )
```

**Forward Pass:**

```python
def forward(
    self,
    input_ids: torch.Tensor,
    task: str,  # "lm", "mlm", or "prediction"
    attention_mask: Optional[torch.Tensor] = None,
    kv_caches: Optional[list[Optional[KVCache]]] = None,
    use_cache: bool = False,
    labels: Optional[torch.Tensor] = None,
) -> ModelOutput
```

**Class Methods:**
- `from_config(config: HyformerConfig) -> Hyformer`: Create model from config
- `from_pretrained(...)`: Load pretrained model from HuggingFace Hub

**Instance Methods:**
- `get_num_params(trainable_only: bool = False) -> int`: Get parameter count
- `save_pretrained(save_directory, safe_serialization=True)`: Save model to disk

### ModelOutput

Output container for model forward pass.

```python
@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    extras: dict[str, Any] | None = None  # Contains hidden_states, kv_caches
```

### ModelInput

Input container supporting dictionary unpacking.

```python
@dataclass
class ModelInput:
    input_ids: torch.Tensor
    task: str
    attention_mask: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    targets: torch.Tensor | None = None
```

## Advanced Usage

### Autoregressive Generation with KV Caching

```python
import torch
from joint_improvement.hyformer import Hyformer
from joint_improvement.hyformer.layers.kv_cache import KVCache

model = Hyformer.from_config(config)
model.eval()

# Initialize KV caches
kv_caches = [None] * len(model.blocks)

# First token
input_ids = torch.randint(0, 32000, (1, 1))  # [batch_size, seq_len=1]
outputs = model(
    input_ids,
    task="lm",
    kv_caches=kv_caches,
    use_cache=True,
)
logits = outputs.logits
kv_caches = outputs.extras["kv_caches"]  # Updated caches

# Subsequent tokens (reuse caches)
next_token = torch.randint(0, 32000, (1, 1))
outputs = model(
    next_token,
    task="lm",
    kv_caches=kv_caches,  # Reuse previous caches
    use_cache=True,
)
```

### Using PredictionHead for Downstream Tasks

For classification and regression tasks, use the `PredictionHead` wrapper:

```python
from joint_improvement.hyformer import Hyformer, HyformerConfig
from joint_improvement.hyformer.layers.prediction import PredictionHead

# Create base model
base_model = Hyformer.from_config(
    HyformerConfig(vocab_size=32000, d_model=512, n_heads=8, n_layers=6)
)

# Wrap with prediction head for binary classification
model = PredictionHead(
    base_model=base_model,
    num_labels=2,  # Binary classification
    d_model=512,
    dropout=0.1,
)

# Forward pass
input_ids = torch.randint(0, 32000, (4, 10))
attention_mask = torch.ones(4, 10)
labels = torch.tensor([0, 1, 0, 1])  # Binary labels

logits, loss = model(
    input_ids,
    task="prediction",
    attention_mask=attention_mask,
    labels=labels,
)

# For regression (num_labels=1)
regression_model = PredictionHead(base_model, num_labels=1, d_model=512)
labels = torch.tensor([[1.0], [2.0], [1.5], [3.0]])
logits, loss = regression_model(
    input_ids,
    task="prediction",
    attention_mask=attention_mask,
    labels=labels,
)
```

### Saving and Loading Models

```python
from joint_improvement.hyformer import Hyformer, HyformerConfig

# Save model
model = Hyformer.from_config(config)
model.save_pretrained("./my_model")  # Saves model.safetensors and config.json

# Load model
loaded_model = Hyformer.from_pretrained("./my_model")

# Save/load checkpoints programmatically
from joint_improvement.hyformer.checkpoint import HyformerCheckpoint

checkpoint = HyformerCheckpoint.from_model(model)
checkpoint.save("./checkpoints/model.safetensors", safe_serialization=True)

# Load checkpoint
checkpoint = HyformerCheckpoint.load("./checkpoints/model.safetensors")
model.load_state_dict(checkpoint.state_dict)
```

### Custom Loss Computation

Losses are computed automatically when labels are provided, but you can compute them manually:

```python
from joint_improvement.hyformer.losses import (
    compute_lm_loss,
    compute_mlm_loss,
    compute_prediction_loss,
)

# Language modeling loss
loss = compute_lm_loss(logits, labels, shift_labels=True)

# Masked language modeling loss with label smoothing
loss = compute_mlm_loss(
    logits,
    labels,
    vocab_size=32000,
    label_smoothing=0.1,
)

# Prediction loss (classification or regression)
loss = compute_prediction_loss(
    logits,
    labels,
    num_labels=10,
    multilabel=False,  # Single-label classification
)
```

## Configuration Files

Configuration files are JSON format:

```json
{
  "vocab_size": 32000,
  "d_model": 512,
  "n_heads": 8,
  "n_layers": 6,
  "max_seq_len": 128,
  "attn_dropout": 0.0,
  "resid_dropout": 0.0,
  "eps": 1e-6
}
```

Load with:

```python
config = HyformerConfig.from_json("path/to/config.json")
```

## Task Modes

### Language Modeling (LM) - Causal
- **Task**: `"lm"`
- **Attention**: Causal (autoregressive)
- **Use Case**: Text generation, next-token prediction
- **Labels**: Next token IDs (will be shifted automatically)

### Masked Language Modeling (MLM) - Bidirectional
- **Task**: `"mlm"`
- **Attention**: Bidirectional (all positions attend to all)
- **Use Case**: BERT-style pretraining
- **Labels**: Original token IDs (only masked positions are used)

### Prediction - Bidirectional
- **Task**: `"prediction"`
- **Attention**: Bidirectional
- **Use Case**: Classification, regression
- **Labels**: Task-specific (class indices for classification, values for regression)

## Model Architecture Details

### Weight Tying

The "lm" and "mlm" heads share weights with the embedding layer:
- Reduces model parameters
- Improves performance
- Standard practice in modern language models

### Initialization

Following LLaMA initialization:
- Embeddings: Normal(0, 0.02)
- Linear layers: Normal(0, 0.02)
- Bias terms: Zeros

### Positional Encoding

- Uses RoPE (Rotary Positional Embeddings)
- Supports variable sequence lengths up to `max_seq_len`
- Efficient for autoregressive generation

## Performance Considerations

- **KV Caching**: Use for autoregressive generation to avoid recomputing previous tokens
- **Batch Size**: Larger batches improve throughput but require more memory
- **Sequence Length**: Quadratic attention complexity - keep sequences as short as possible
- **Mixed Precision**: Use `torch.amp` for faster training with minimal accuracy loss

## Examples

See the project examples directory for complete training scripts and use cases.

## License

See the main project LICENSE file.

## Citation

If you use Hyformer in your research, please cite:

```bibtex
@software{hyformer,
  title={Hyformer: A Hybrid Transformer for Multi-Task Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/joint-improvement}
}
```

## Contributing

Contributions are welcome! Please see the main project CONTRIBUTING guide.

## Support

For issues and questions, please open an issue on the project repository.

