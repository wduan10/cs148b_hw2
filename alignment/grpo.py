from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
) -> dict[str, Tensor]:
    """Tokenize prompt/output pairs and build a response mask over the labels."""
    # Tokenize each prompt and output separately (no special tokens).
    prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    output_ids = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]

    full_sequences = [p + o for p, o in zip(prompt_ids, output_ids)]
    # Sequence length after the causal shift (drop one token from each end).
    seq_lens = [len(s) - 1 for s in full_sequences]
    max_len = max(seq_lens)

    pad_id = tokenizer.pad_token_id

    batch_input_ids = []
    batch_labels = []
    batch_response_mask = []

    for full_seq, p_ids, o_ids in zip(full_sequences, prompt_ids, output_ids):
        seq_len = len(full_seq) - 1
        pad_len = max_len - seq_len

        # input_ids: all tokens except last, right-padded.
        inp = full_seq[:-1] + [pad_id] * pad_len
        # labels: all tokens except first (shifted), right-padded.
        lbl = full_seq[1:] + [pad_id] * pad_len

        # response_mask is True only on response token positions in the labels tensor.
        # After the shift, prompt occupies positions [0, prompt_len - 1) in labels,
        # and the response occupies positions [prompt_len - 1, prompt_len - 1 + response_len).
        prompt_len = len(p_ids)
        response_len = len(o_ids)
        mask = (
            [False] * (prompt_len - 1)
            + [True] * response_len
            + [False] * pad_len
        )

        batch_input_ids.append(inp)
        batch_labels.append(lbl)
        batch_response_mask.append(mask)

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
        "response_mask": torch.tensor(batch_response_mask, dtype=torch.bool),
    }


def compute_entropy(logits: Tensor) -> Tensor:
    """Compute per-token entropies over the vocabulary dimension."""
    # Numerically stable: log_softmax uses logsumexp internally.
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """Score conditional log-probabilities for a batch of prompt/response examples."""
    logits = model(input_ids).logits  # (batch, seq_len, vocab)
    log_probs = torch.log_softmax(logits, dim=-1)
    # Gather the log-prob of each label token.
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    out: dict[str, Tensor] = {"log_probs": token_log_probs}
    if return_token_entropy:
        out["token_entropy"] = compute_entropy(logits)
    return out


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> Tensor:
    """Sum over masked elements and normalize by the provided constant."""
    masked = tensor * mask
    if dim is None:
        return masked.sum() / normalize_constant
    return masked.sum(dim=dim) / normalize_constant


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Compute raw rewards and per-group normalized advantages for GRPO."""
    raise NotImplementedError


def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the per-token GRPO-Clip loss."""
    raise NotImplementedError


def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    advantages: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Backpropagate a single GRPO microbatch loss."""
    raise NotImplementedError


def log_generations(
    prompts: Sequence[str],
    responses: Sequence[str],
    ground_truths: Sequence[str],
    reward_infos: Sequence[dict[str, float]],
    token_entropies: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Create serializable generation logs for debugging training runs."""
    records = []
    response_lengths: list[int] = []
    correct_lengths: list[int] = []
    incorrect_lengths: list[int] = []

    for i, (prompt, response, ground_truth, reward_info) in enumerate(
        zip(prompts, responses, ground_truths, reward_infos, strict=True)
    ):
        resp_len = len(response.split())
        is_correct = float(reward_info.get("answer_reward", 0.0)) == 1.0
        response_lengths.append(resp_len)
        (correct_lengths if is_correct else incorrect_lengths).append(resp_len)

        record: dict[str, Any] = {
            "idx": i,
            "prompt": prompt,
            "response": response,
            "ground_truth": ground_truth,
            "reward": reward_info,
            "response_length": resp_len,
        }
        if token_entropies is not None:
            record["avg_token_entropy"] = float(token_entropies[i])

        records.append(record)

    def _mean(lst: list[int]) -> float | None:
        return sum(lst) / len(lst) if lst else None

    summary: dict[str, Any] = {
        "avg_response_length": _mean(response_lengths),
        "avg_correct_response_length": _mean(correct_lengths),
        "avg_incorrect_response_length": _mean(incorrect_lengths),
        "n_correct": len(correct_lengths),
        "n_total": len(records),
    }
    if token_entropies is not None:
        summary["avg_token_entropy"] = (
            sum(float(e) for e in token_entropies) / len(token_entropies)
            if token_entropies else None
        )

    for record in records:
        record["summary"] = summary

    return records


def train_grpo(*args, **kwargs) -> dict[str, Any]:
    """Run the full GRPO training loop from Section 3.5."""
    raise NotImplementedError
