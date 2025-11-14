import numpy as np
import torch
from torch.nn.functional import log_softmax


class BaseSearchMixin:
    """Base class for search-based generation."""

    def _to_numpy(self, logits: torch.Tensor) -> np.ndarray:
        if logits.is_cuda:
            logits = logits.detach()
        return logits.cpu().numpy()

    def _compute_log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return log_softmax(logits, dim=-1)

    def _scale_logits(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        return logits / temperature

    def _crop_logits(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("Inf")
        return logits
