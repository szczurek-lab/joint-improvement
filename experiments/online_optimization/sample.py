from collections.abc import Callable, Sequence

import torch

from joint_improvement.utils.chemistry import calculate_validity, has_radicals


def oracle_fn(
    input_idx: torch.Tensor,
    tokenizer: "SMILESTokenizer",
    target: str,
    model: torch.nn.Module,
    model_device: torch.device,
) -> list[float]:
    # TODO: implement guards, including low perplexity, etc.
    smiles = tokenizer.decode(input_idx)
    is_valid = calculate_validity(smiles) & (not has_radicals(smiles))
    if not is_valid:
        return 0.0

    input_idx = input_idx.unsqueeze(0).to(model_device)
    attention_mask = torch.ones_like(input_idx).to(model_device)
    prediction = model.predict(input_ids=input_idx, attention_mask=attention_mask)[:, 0]  # choose only docking score prediction

    return prediction.item()


def sample_new_solutions(
    tokenizer: "SMILESTokenizer",
    model: torch.nn.Module,
    advantage_fn: Callable,
    oracle: Callable,
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
) -> Sequence[str]:
    """Samples new solutions from the model."""
    prefix_input_ids = (
        torch.tensor([tokenizer.task_token_ids["lm"], tokenizer.bos_token_id])
        .unsqueeze(0)
        .to(device)
    )

    sampled_solutions: list[str] = []
    while True:
        if len(sampled_solutions) >= num_samples:
            break

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
        )
        smiles = tokenizer.batch_decode(samples, skip_special_tokens=True)
        valid_smiles = [
            smile
            for smile in smiles
            if calculate_validity(smile) & (not has_radicals(smile))
        ]  # SATURN-style safety guards ensuring comparability

        # TODO: add diversity filtering

        sampled_solutions.extend(valid_smiles)
        sampled_solutions = list(set(sampled_solutions))


    return sampled_solutions[-num_samples:]
