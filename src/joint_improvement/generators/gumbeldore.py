"""Gumbeldore sampling implementation for sequence generation.

This implementation is based on the Gumbeldore sampling algorithm, which uses
stochastic beam search with advantage-weighted sampling for conditional generation.

References
----------
  Official Gumbeldore implementation: https://github.com/grimmlab/gumbeldore
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any, TypeAlias

import torch

from .generator import GeneratorMixin
from .utils.gumbeldore.incremental_sbs import IncrementalSBS

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from joint_improvement.hyformer.model import Hyformer

    from .utils.gumbeldore.stochastic_beam_search import BeamLeaf, State

_LOGGER = logging.getLogger(__name__)
StateList: TypeAlias = list[torch.Tensor]
StateTensor: TypeAlias = torch.Tensor


class GumbeldoreMixin(GeneratorMixin):
    """Gumbeldore sampling-based generation."""

    def _cast_to_tensor(self, states: StateList) -> torch.Tensor:
        return torch.stack(states, dim=0)

    def _cast_to_states(self, tensor: torch.Tensor) -> StateList:
        return [tensor[batch_idx].squeeze(0) for batch_idx in range(tensor.shape[0])]

    def _build_fn(self, fn: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
        return functools.partial(fn, **kwargs)

    def _child_log_probability_fn(
        self, states: StateList, temperature: float = 1.0, top_k: int | None = None
    ) -> list[np.ndarray]:
        logits = self._get_model_logits(self._cast_to_tensor(states))
        if temperature != 1.0:
            logits = self._scale_logits(logits, temperature=temperature)
        if top_k is not None:
            logits = self._apply_top_k(logits, top_k=top_k)
        log_probs = self._compute_log_probs(logits)
        output_states: StateList = self._cast_to_states(log_probs.cpu().numpy())
        return output_states

    def _child_transition_fn(
        self,
        state_action_pairs: list[tuple[StateTensor, int]],
        max_sequence_length: int,
        eos_token_id: int,
        pad_token_id: int | None = None,
    ) -> list[tuple[StateTensor, bool]]:
        assert eos_token_id is not None, "EOS token ID must be provided."
        assert isinstance(eos_token_id, int), "EOS token ID must be an integer."
        new_states: list[tuple[StateTensor, bool]] = []
        for prefix, action in state_action_pairs:
            new_ids = torch.cat([prefix, torch.tensor([action], dtype=torch.long, device=prefix.device)], dim=0)
            is_leaf = (int(action) == int(eos_token_id)) or (new_ids.size(0) >= max_sequence_length)
            new_states.append((new_ids, bool(is_leaf)))
        return new_states

    def _leaf_evaluation_fn(
        self,
        input_ids: torch.LongTensor,
        advantage_fn: Callable[[float], float],
        oracle_fn: Callable[[torch.LongTensor], float] | None = None,
    ) -> float:
        objective_fn_evaluation = (
            self._get_model_predictions(input_ids=input_ids) if oracle_fn is None else oracle_fn(input_ids)  # type: ignore[attr-defined]
        )
        return advantage_fn(float(objective_fn_evaluation))

    def _get_sampler(
        self,
        initial_states: StateList,
        child_log_probability_fn: Callable[[StateList], list[np.ndarray]],
        child_transition_fn: Callable[[list[tuple[StateTensor, int]]], list[tuple[StateTensor, bool]]],
        leaf_evaluation_fn: Callable[[StateTensor], float],
    ) -> IncrementalSBS:
        return IncrementalSBS(
            initial_states=initial_states,
            child_log_probability_fn=child_log_probability_fn,
            child_transition_fn=child_transition_fn,
            leaf_evaluation_fn=leaf_evaluation_fn,
            memory_aggressive=False,
        )

    def generate(
        self,
        prefix_input_ids: torch.LongTensor,
        max_sequence_length: int,
        advantage_fn: Callable[[float], float],
        eos_token_id: int,
        oracle_fn: Callable[[StateTensor], float] | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        beam_width: int = 32,
        num_rounds: int = 4,
        advantage_constant: float = 1.0,
        normalize_advantage_value: bool = True,
        min_nucleus_top_p: float = 1.0,
    ) -> torch.FloatTensor:
        """Generate sequences using incremental stochastic beam search.

        This method performs conditional sequence generation using the incremental stochastic beam search (SBS) approach
        and Gumbeldore-style log probability updates.

        Parameters
        ----------
        prefix_input_ids : torch.LongTensor
            Input token IDs tensor representing the initial sequence/prompt of shape
            (sequence_length, ) for a single sequence.
        max_sequence_length : int
            Maximum sequence length of the generated sequence.
        advantage_fn : Callable[[float], float]
            Function that calculates the advantage of a given prediction to be maximized.
            The function should accept a float value and return a float value.
            Example: lambda x: (x - target_value) ** 2.
        eos_token_id : int
            EOS token ID.
        oracle_fn : Optional[Callable[[torch.LongTensor], float]]
            Function that evaluates the leaf states (complete sequences) and returns a float
            score. The function should accept a torch.LongTensor object and return a float value.
            If None, the model predictions are used to calculate the advantage.
        temperature : float
            Temperature for the logits.
        top_k : Optional[int]
            Top-k value for the logits.
        beam_width : int
            Beam width for one round of SBS.
        num_rounds : int
            Number of SBS rounds, where we update the log-probs after each round.
        advantage_constant : float
            Constant for advantage calculation in Gumbeldore.
        normalize_advantage_value : bool
            Min-max normalize advantage values, to set the highest/lowest advantage to 1/-1.
        min_nucleus_top_p : float
            Minimum nucleus (top-p) sampling parameter.

        Returns
        -------
        List[torch.LongTensor]
            List of generated sequences of shape (sequence_length,) for each round.

        Raises
        ------
        AttributeError
            If required instance attributes are not set.
        RuntimeError
            If model logits cannot be computed or sampling fails.

        Notes
        -----
        Requires a `_get_model_logits` method to be implemented in the class.
        """
        child_log_probability_fn = self._build_fn(self._child_log_probability_fn, temperature=temperature, top_k=top_k)

        child_transition_fn = self._build_fn(
            self._child_transition_fn,
            max_sequence_length=max_sequence_length,
            eos_token_id=eos_token_id,
        )

        leaf_evaluation_fn = self._build_fn(self._leaf_evaluation_fn, advantage_fn=advantage_fn, oracle_fn=oracle_fn)

        _sampler = self._get_sampler(
            initial_states=self._cast_to_states(prefix_input_ids.unsqueeze(0)),  # equivalent to batch size == 1
            child_log_probability_fn=child_log_probability_fn,
            child_transition_fn=child_transition_fn,
            leaf_evaluation_fn=leaf_evaluation_fn,
        )

        result = _sampler.perform_incremental_sbs(
            beam_width=beam_width,
            num_rounds=num_rounds,
            log_prob_update_type="gumbeldore",
            advantage_constant=advantage_constant,
            min_max_normalize_advantage=normalize_advantage_value,
            expected_value_use_simple_mean=False,
            use_pure_outcomes=False,
            normalize_advantage_by_visit_count=False,
            perform_first_round_deterministic=False,
            min_nucleus_top_p=min_nucleus_top_p,
            return_round_info=False,
        )

        return self._cast_incremental_sbs_result(result)

    def _cast_incremental_sbs_result(self, result: list[list[BeamLeaf]]) -> list[list[State]]:
        output_result = result[0]
        return [leaf.state for leaf in output_result]

    @torch.inference_mode()
    def _get_model_logits(self: Hyformer, input_ids: torch.LongTensor) -> torch.FloatTensor:  # type: ignore[misc]
        logits = self.forward(input_ids=input_ids, task="lm")["logits"]  # type: ignore[attr-defined]
        return logits[:, [-1]]  # next token logits


class GumbeldoreMixinV1(GumbeldoreMixin):  # type: ignore[misc]
    """Gumbeldore sampling-based generation for V1 models (legacy)."""

    @torch.inference_mode()
    def _get_model_logits(self: Hyformer, input_ids: torch.LongTensor) -> torch.FloatTensor:  # type: ignore[misc]
        """Return logits for the next token of shape (batch_size, 1, vocab_size)."""
        output = self.forward(input_ids=input_ids, attention_mask=None, task="lm", use_cache=False)
        logits = output["logits"]
        return logits[:, [-1]]

    @torch.inference_mode()
    def _get_model_predictions(self: Hyformer, input_ids: torch.LongTensor) -> float:  # type: ignore[misc]
        """Return predictions for the downstream task, e.g., of shape (batch_size, num_tasks) for regression."""
        input_ids = input_ids.unsqueeze(0)  # equivalent to batch size == 1
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        return self.predict(input_ids=input_ids, attention_mask=attention_mask).item()  # type: ignore[attr-defined]


class GumbeldoreMixinV2(GumbeldoreMixin):  # type: ignore[misc]
    """Gumbeldore sampling-based generation for V2 models (legacy)."""

    @torch.inference_mode()
    def _get_model_logits(self: Hyformer, input_ids: torch.LongTensor) -> torch.FloatTensor:  # type: ignore[misc]
        logits = self.forward(input_ids=input_ids, attention_mask=None, task="lm", use_cache=False)["logits"]
        return logits[:, [-1]]

    @torch.inference_mode()
    def _get_model_predictions(  # type: ignore[misc]
        self: Hyformer,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        _logits_prediction = self.forward(
            input_ids=input_ids, attention_mask=attention_mask, task="prediction", **kwargs
        )["logits"]
        if self.prediction_task_type == "classification":  # type: ignore[attr-defined]
            return torch.sigmoid(_logits_prediction)
        elif self.prediction_task_type == "regression":  # type: ignore[attr-defined]
            return _logits_prediction
        else:
            raise ValueError("Variable `prediction_task_type` must be either `classification` or `regression`.")
