from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from joint_improvement.utils.chemistry import calculate_validity, has_radicals
from joint_improvement.utils.data.target_transforms import ZScoreScaler

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from joint_improvement.tokenizers.smiles import SMILESTokenizer

from experiments.offline_optimization.helpers import (
    calculate_model_predictions,
    manual_docking_target_transform,
)


def _model_predictions_to_scores(predictions: torch.Tensor) -> float:
    # predictions = manual_docking_target_transform(
    #     predictions
    # )  
    ds_predictions = (predictions[:, 0] - 8.19646667) / 1.02184514
    ds_predictions = (-1) * ds_predictions
    ds_predictions = (ds_predictions + 1) / 2
    sa_predictions = (predictions[:, 1] - 3.06933226) / 0.84075993
    sa_predictions = (-1) * sa_predictions
    sa_predictions = (sa_predictions + 1) / 2
    qed_predictions = (predictions[:, 2] - 0.73222901) / 0.14083078
    qed_predictions = (qed_predictions + 1) / 2
    return (ds_predictions + sa_predictions + qed_predictions) / 3

# return (ds_predictions + sa_predictions + qed_predictions) / 3

def oracle_fn(
    input_idx: torch.Tensor,
    tokenizer: SMILESTokenizer,
    target_transforms: list[ZScoreScaler],
    model: torch.nn.Module,
    device: torch.device,
) -> float:
    smiles = tokenizer.decode(input_idx)
    # if not (calculate_validity(smiles) & (not has_radicals(smiles))):
    #     return 0.0

    input_idx = input_idx.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_idx).to(device)
    predictions = (
        model.predict(
            input_ids=input_idx,
            attention_mask=attention_mask,
        )
        .detach()
        .cpu()
    )
    for target_transform in target_transforms:
        predictions = target_transform.inverse_transform(predictions)
        
    # if predictions[:, 1].item() > 5:
    #     return 0.0
    
    # if predictions[:, 2].item() < 0.5:
    #     return 0.0
    
    # if predictions[:, 1].item() > 5:
    #     return 0.0
    
    return _model_predictions_to_scores(predictions).item()


def sample_solutions(
    tokenizer: SMILESTokenizer,
    model: torch.nn.Module,
    advantage_fn: Callable,
    oracle_fn: Callable,
    device: torch.device,
    num_samples: int,
    max_sequence_length: int,
    temperature: float,
    top_k: int | None,
    beam_width: int,
    num_rounds: int,
    advantage_constant: float,
    normalize_advantage_value: bool,
    min_nucleus_top_p: float,
    target_transforms: list[ZScoreScaler],
    rng: np.random.Generator,
) -> tuple[Sequence[str], Sequence[float]]:
    """Samples new solutions from the model and returns tuples of (SMILES, ranking_score)."""
    prefix_input_ids = (
        torch.tensor([tokenizer.task_token_ids["lm"], tokenizer.bos_token_id])
        .unsqueeze(0)
        .to(device)
    )
    
    oracle = partial(oracle_fn, tokenizer=tokenizer, target_transforms=target_transforms, model=model, device=device)

    sampled_solutions: list[str] = []
    sampled_scores: list[float] = []

    with tqdm(total=num_samples, desc="Sampling new solutions") as pbar:
        while True:
            if len(sampled_solutions) >= num_samples:
                return sampled_solutions[:num_samples], sampled_scores[:num_samples]

            samples = model.generate(
                prefix_input_ids=prefix_input_ids,
                max_sequence_length=max_sequence_length - len(prefix_input_ids[0]),
                advantage_fn=advantage_fn,
                eos_token_id=tokenizer.eos_token_id,
                oracle_fn=oracle,
                temperature=temperature,
                top_k=top_k,
                beam_width=beam_width,
                num_rounds=num_rounds,
                advantage_constant=advantage_constant,
                normalize_advantage_value=normalize_advantage_value,
                min_nucleus_top_p=min_nucleus_top_p,
                rng=rng,
            )
            samples = samples[-beam_width:]  # take last beam_width of solutions
            smiles = tokenizer.batch_decode(samples, skip_special_tokens=True)
            valid_smiles = [
                smile
                for smile in smiles
                if calculate_validity(smile) & (not has_radicals(smile))
            ]  # SATURN-style safety guards ensuring comparability
            if len(valid_smiles) == 0:
                continue

            # TODO: add diversity filtering

            predictions = calculate_model_predictions(
                valid_smiles,
                tokenizer,
                model,
                device,
                target_transforms=target_transforms,
            )  # raw predictions
            scores = _model_predictions_to_scores(predictions)
            best_idx = np.argmax(
                scores
            )  # choose the lowest docking score (best)
            best_solution = valid_smiles[best_idx]
            best_score = scores[best_idx]
            if best_solution not in sampled_solutions:
                sampled_solutions.append(best_solution)
                sampled_scores.append(best_score)
            else:
                continue
            pbar.update(1)

