# model.py
from __future__ import annotations

import torch
from torch import nn

from joint_improvement.generators import UnconditionalGeneratorMixin

from .config import HyformerConfig
from .layers.kv_cache import KVCache
from .layers.prediction_head import PredictionHeadModule
from .layers.transformer_layer import TransformerBlock
from .losses import compute_lm_loss, compute_mlm_loss, compute_prediction_loss
from .outputs import ModelOutput
from .pretrained import PretrainedMixin


class Hyformer(PretrainedMixin, UnconditionalGeneratorMixin, nn.Module):
    """
    Hyformer backbone model.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary (number of tokens).
    d_model : int
        Model dimension (hidden size).
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of decoder blocks (transformer layers).
    max_seq_len : int
        Maximum sequence length for positional encodings and attention masks.

    Attributes
    ----------
    vocab_size : int
        Size of the vocabulary.
    num_prediction_tasks : int | None
        Number of prediction tasks (if prediction head is enabled).
    embed : nn.Embedding
        Token embedding layer of shape [vocab_size, d_model].
    blocks : nn.ModuleList
        List of decoder blocks, each containing attention and MLP sub-layers.
    final_norm : nn.LayerNorm
        Final layer normalization applied before the task heads.
    heads : nn.ModuleDict
        Task-specific heads. Contains "lm", "mlm", and optionally "prediction".
        The "lm" and "mlm" heads are weight-tied with the embedding layer.
        The "prediction" head is a PredictionHeadModule following DINOv2 best practices
        (LayerNorm, dropout, classifier) and uses the CLS token representation.

    Notes
    -----
    The model uses weight tying between the embedding and language modeling head,
    which reduces the number of parameters and can improve performance.

    The forward pass processes input tokens through:
        1. Token embeddings
        2. Stack of transformer blocks (with optional KV caching)
        3. Final layer normalization

    The model supports both causal (autoregressive) and bidirectional attention
    modes, controlled by the `is_causal` parameter in the forward pass.

    The model includes UnconditionalGeneratorMixin by default, providing standard
    autoregressive generation methods (greedy, top-k, top-p, temperature sampling)
    via the `generate()` method.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int,
        num_prediction_tasks: int | None = None,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_prediction_tasks = num_prediction_tasks

        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    max_seq_len=max_seq_len,
                    attn_dropout=attn_dropout,
                    resid_dropout=resid_dropout,
                    eps=eps,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model, eps=eps)
        self.heads = nn.ModuleDict(
            {"lm": nn.Linear(d_model, vocab_size, bias=False), "mlm": nn.Linear(d_model, vocab_size, bias=False)}
        )
        # Weight tying: both heads share embedding weights
        self.heads["lm"].weight = self.embed.weight
        self.heads["mlm"].weight = self.embed.weight

        if num_prediction_tasks is not None:
            self.heads["prediction"] = PredictionHeadModule(
                d_model=d_model,
                num_labels=num_prediction_tasks,
                dropout=0.1,
            )

        # Apply LLaMA-specific weight initialization
        self._apply_llama_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        task: str,
        attention_mask: torch.Tensor | None = None,
        kv_caches: list[KVCache | None] | None = None,
        use_cache: bool = False,
        labels: torch.Tensor | None = None,
    ) -> ModelOutput:
        """
        Forward pass through the model.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs of shape [B, T] where B is batch size and
            T is sequence length.
        task : str
            Task name (e.g., "lm", "mlm", "prediction"). Used to determine
            attention type automatically.
        attention_mask : Optional[torch.Tensor], default=None
            Padding mask tensor of shape [B, T] where 1 indicates valid tokens
            and 0 indicates padding. Used to create attention mask for bidirectional tasks.
        kv_caches : Optional[list[Optional[KVCache]]], default=None
            Optional list of key-value caches, one per decoder block.
            If None and use_cache=True, will be initialized as None for
            each block. If provided, should have length equal to n_layers.
        use_cache : bool, default=False
            Whether to use and update key-value caches for autoregressive
            generation.
        labels : Optional[torch.Tensor], default=None
            Optional labels for computing loss. If provided, loss will be computed.

        Returns
        -------
        ModelOutput
            Model output containing:
            - logits: Tensor of shape [B, T, vocab_size] for LM/MLM tasks,
                     or [B, num_prediction_tasks] for prediction tasks
            - loss: Optional loss tensor if labels provided
            - extras: Dict containing hidden_states and kv_caches

        Notes
        -----
        For prediction tasks, the model extracts the CLS token (first token) from
        the hidden states and passes it through the PredictionHeadModule, which
        applies LayerNorm, dropout, and classification following DINOv2 best practices.
        """
        if task not in self.heads:
            raise ValueError(f"Unknown task '{task}'. Available heads: {list(self.heads.keys())}")

        is_causal = task == "lm"

        B, T = input_ids.shape
        x = self.embed(input_ids)

        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)

        # Build attention mask for bidirectional attention
        if is_causal:
            attn_mask = None
        else:
            if attention_mask is None:
                attn_mask = None
            else:
                # Convert [B, T] padding mask (1=keep, 0=pad) -> SDPA bool mask [B, 1, 1, T] (True=mask)
                attn_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)

        for i, block in enumerate(self.blocks):
            x, kv_caches[i] = block(
                x,
                attn_mask=attn_mask,
                kv_cache=kv_caches[i],
                use_cache=use_cache,
                is_causal=is_causal,
            )

        x = self.final_norm(x)

        # Compute logits from task-specific head
        if task == "prediction":
            # For prediction tasks, use CLS token (first token) - DINOv2 best practice
            # Ensure contiguous tensor to avoid compilation issues when switching tasks
            pooled = x[:, 0, :].contiguous()  # [B, d_model]
            logits = self.heads[task](pooled)  # [B, num_labels]
        else:
            # For LM/MLM tasks, use full sequence
            # Ensure contiguous tensor for consistent compilation behavior
            logits = self.heads[task](x.contiguous())  # [B, T, vocab_size]

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            if task == "lm":
                loss = compute_lm_loss(logits, labels, shift_labels=is_causal)
            elif task == "mlm":
                loss = compute_mlm_loss(logits, labels)
            elif task == "prediction":
                if self.num_prediction_tasks is None:
                    raise ValueError("num_prediction_tasks must be set for prediction task")
                loss = compute_prediction_loss(logits, labels, num_labels=self.num_prediction_tasks)

        # Store hidden states and kv_caches in extras
        extras = {
            "hidden_states": x,
            "kv_caches": kv_caches,
        }

        return ModelOutput(
            logits=logits,
            loss=loss,
            extras=extras,
        )

    def _apply_llama_init(self) -> None:
        """Apply LLaMA-specific weight initialization.

        Initialization scheme following LLaMA:
        - Embeddings: Normal(0, 0.02)
        - Linear layers: Normal(0, 0.02)
        - RMSNorm weights: Already initialized to ones (default in RMSNorm)
        - LayerNorm: Default PyTorch initialization (already applied)

        Note: Since the "lm" and "mlm" heads are weight-tied to embed.weight,
        initializing embed.weight also initializes those heads automatically.
        """
        # Initialize embeddings with normal distribution (std=0.02)
        # This also initializes weight-tied heads ("lm" and "mlm") since they share weights
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        # Initialize all other linear layers with normal distribution
        # Skip weight-tied heads (already initialized via embed.weight)
        # Skip prediction head classifier (initialized by PredictionHeadModule._init_weights())
        tied_heads = {self.heads["lm"], self.heads["mlm"]}
        prediction_head_classifier = None
        if hasattr(self, "heads") and "prediction" in self.heads:
            prediction_head_classifier = self.heads["prediction"].classifier

        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Skip weight-tied heads (already initialized via embed.weight)
                if module in tied_heads:
                    continue
                # Skip prediction head classifier (initialized by PredictionHeadModule itself)
                if module is prediction_head_classifier:
                    continue
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_num_params(self, trainable_only: bool = False) -> int:
        """Get the number of parameters in the model.

        Parameters
        ----------
        trainable_only : bool, default=False
            If True, only count trainable parameters. If False, count all parameters.

        Returns
        -------
        int
            Number of parameters in the model.

        Examples
        --------
        >>> model = Hyformer.from_config(config)
        >>> total_params = model.get_num_params()
        >>> trainable_params = model.get_num_params(trainable_only=True)
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, config: HyformerConfig) -> Hyformer:
        """Create Hyformer model from configuration.

        Parameters
        ----------
        config : HyformerConfig
            Model configuration.

        Returns
        -------
        Hyformer
            Instantiated Hyformer model.

        Examples
        --------
        >>> config = HyformerConfig.from_pretrained("configs/hyformer/base.json")
        >>> model = Hyformer.from_config(config)
        """
        # Only pass parameters that Hyformer.__init__() accepts
        config_dict = config.to_dict()
        return cls(
            vocab_size=int(config_dict["vocab_size"]),
            d_model=int(config_dict["d_model"]),
            n_heads=int(config_dict["n_heads"]),
            n_layers=int(config_dict["n_layers"]),
            max_seq_len=int(config_dict["max_seq_len"]),
            num_prediction_tasks=int(config_dict["num_prediction_tasks"])
            if config_dict.get("num_prediction_tasks") is not None
            else None,
            attn_dropout=config_dict.get("attn_dropout", 0.0),
            resid_dropout=config_dict.get("resid_dropout", 0.0),
            eps=config_dict.get("eps", 1e-6),
        )
