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
    reward_infos = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths, strict=True)
    ]
    raw_rewards = torch.tensor(
        [float(info["reward"]) for info in reward_infos],
        dtype=torch.float32,
    )

    # Reshape into (n_groups, group_size) for group-wise statistics.
    grouped = raw_rewards.view(-1, group_size)
    group_means = grouped.mean(dim=1, keepdim=True)
    centered = grouped - group_means

    if normalize_by_std:
        group_stds = grouped.std(dim=1, keepdim=True, unbiased=False)
        advantages = centered / (group_stds + advantage_eps)
    else:
        advantages = centered

    advantages = advantages.reshape(-1)

    metadata: dict[str, float] = {
        "mean_reward": float(raw_rewards.mean()),
        "std_reward": float(raw_rewards.std(unbiased=False)),
        "max_reward": float(raw_rewards.max()),
        "min_reward": float(raw_rewards.min()),
        "mean_format_reward": float(sum(info.get("format_reward", 0.0) for info in reward_infos) / len(reward_infos)),
        "mean_answer_reward": float(sum(info.get("answer_reward", 0.0) for info in reward_infos) / len(reward_infos)),
    }

    return advantages, raw_rewards, metadata


def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the per-token GRPO-Clip loss."""
    # Probability ratio π_θ / π_θ_old in log-space for numerical stability.
    ratios = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratios = torch.clamp(ratios, 1.0 - cliprange, 1.0 + cliprange)

    # Broadcast advantages from (batch, 1) over sequence length.
    broadcast_advantages = advantages.expand_as(policy_log_probs)

    unclipped = ratios * broadcast_advantages
    clipped   = clipped_ratios * broadcast_advantages

    loss = -torch.minimum(unclipped, clipped)

    # Track which tokens were clipped for logging.
    is_clipped = (clipped < unclipped).to(policy_log_probs.dtype)
    metadata: dict[str, Tensor] = {
        "clip_fraction": is_clipped,
    }

    return loss, metadata


def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    advantages: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Backpropagate a single GRPO microbatch loss."""
    per_token_loss, clip_metadata = compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # Mask out non-response tokens and average over each example's response tokens.
    mask = response_mask.to(per_token_loss.dtype)
    per_example_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1)

    # Average over the batch, then scale down for gradient accumulation.
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()

    metadata: dict[str, Tensor] = {
        **clip_metadata,
        "per_example_loss": per_example_loss.detach(),
    }
    return loss.detach(), metadata


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


def train_grpo(
    policy: torch.nn.Module,
    tokenizer,
    vllm_model,
    reward_fn: Callable[[str, str], dict[str, float]],
    train_examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    prompts_train: list[str],
    prompts_val: list[str],
    *,
    # Hyperparameters with PDF-recommended defaults.
    n_grpo_steps: int = 50,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 32,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 256,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 32,
    gradient_accumulation_steps: int = 16,
    cliprange: float = 1.0,
    normalize_by_std: bool = True,
    val_interval: int = 5,
    grad_clip: float = 1.0,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    """Run the full GRPO training loop from Section 3.5."""
    from importlib import import_module
    vllm_mod = import_module("vllm")

    assert train_batch_size % gradient_accumulation_steps == 0
    assert rollout_batch_size % group_size == 0
    assert train_batch_size >= group_size

    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    n_prompts_per_rollout_batch = rollout_batch_size // group_size

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    rollout_sampling_params = vllm_mod.SamplingParams(
        temperature=sampling_temperature,
        top_p=1.0,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    val_sampling_params = vllm_mod.SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    history: dict[str, Any] = {"train": [], "val": []}
    import random

    for step in range(n_grpo_steps):
        policy.eval()

        # ── 1. Sample a rollout batch ─────────────────────────────────────────
        batch_indices = random.sample(range(len(prompts_train)), n_prompts_per_rollout_batch)
        batch_prompts = [prompts_train[i] for i in batch_indices]
        batch_ground_truths = [train_examples[i]["ground_truth"] for i in batch_indices]

        # Repeat each prompt group_size times for vLLM.
        repeated_prompts = [p for p in batch_prompts for _ in range(group_size)]
        repeated_ground_truths = [gt for gt in batch_ground_truths for _ in range(group_size)]

        rollout_outputs = vllm_model.generate(repeated_prompts, rollout_sampling_params)
        rollout_responses = [out.outputs[0].text for out in rollout_outputs]

        # ── 2. Compute group-normalised advantages ───────────────────────────
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=normalize_by_std,
        )
        advantages = advantages.to(device)

        # ── 3. Tokenize rollouts and cache old log-probs (no grad) ──────────
        tokenized = tokenize_prompt_and_output(
            prompt_strs=repeated_prompts,
            output_strs=rollout_responses,
            tokenizer=tokenizer,
        )
        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        response_mask = tokenized["response_mask"].to(device)

        with torch.no_grad():
            old_log_prob_dict = get_response_log_probs(
                model=policy,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=False,
            )
        old_log_probs = old_log_prob_dict["log_probs"].detach()

        # ── 4. Gradient update epochs ────────────────────────────────────────
        step_losses: list[float] = []
        step_grad_norms: list[float] = []

        for _epoch in range(epochs_per_rollout_batch):
            # Shuffle rollout batch indices for each epoch.
            perm = torch.randperm(rollout_batch_size)
            n_microbatches = rollout_batch_size // micro_train_batch_size

            optimizer.zero_grad()
            for mb_idx in range(n_microbatches):
                mb_indices = perm[mb_idx * micro_train_batch_size:(mb_idx + 1) * micro_train_batch_size]

                mb_input_ids = input_ids[mb_indices]
                mb_labels = labels[mb_indices]
                mb_response_mask = response_mask[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices].unsqueeze(1)

                policy.train()
                policy_log_prob_dict = get_response_log_probs(
                    model=policy,
                    input_ids=mb_input_ids,
                    labels=mb_labels,
                    return_token_entropy=False,
                )
                mb_policy_log_probs = policy_log_prob_dict["log_probs"]

                mb_loss, _mb_meta = grpo_microbatch_train_step(
                    policy_log_probs=mb_policy_log_probs,
                    response_mask=mb_response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    advantages=mb_advantages,
                    old_log_probs=mb_old_log_probs,
                    cliprange=cliprange,
                )
                step_losses.append(float(mb_loss))

            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()
            step_grad_norms.append(float(grad_norm))

        train_log: dict[str, Any] = {
            "step": step,
            "mean_loss": sum(step_losses) / max(len(step_losses), 1),
            "mean_grad_norm": sum(step_grad_norms) / max(len(step_grad_norms), 1),
            **reward_meta,
        }
        history["train"].append(train_log)
        print(
            f"[step {step:3d}] loss={train_log['mean_loss']:.4f} "
            f"grad_norm={train_log['mean_grad_norm']:.4f} "
            f"reward={reward_meta['mean_reward']:.4f} "
            f"answer_reward={reward_meta['mean_answer_reward']:.4f}"
        )

        # ── 5. Periodic validation ───────────────────────────────────────────
        if (step + 1) % val_interval == 0 or step == n_grpo_steps - 1:
            policy.eval()
            val_subset = val_examples[:256]
            val_prompts_subset = prompts_val[:256]
            val_gts = [ex["ground_truth"] for ex in val_subset]

            val_outputs = vllm_model.generate(val_prompts_subset, val_sampling_params)
            val_responses = [out.outputs[0].text for out in val_outputs]
            val_reward_infos = [reward_fn(r, gt) for r, gt in zip(val_responses, val_gts)]

            val_mean_answer_reward = sum(
                float(ri["answer_reward"]) for ri in val_reward_infos
            ) / len(val_reward_infos)
            val_mean_reward = sum(
                float(ri["reward"]) for ri in val_reward_infos
            ) / len(val_reward_infos)

            val_log: dict[str, Any] = {
                "step": step,
                "val_mean_reward": val_mean_reward,
                "val_mean_answer_reward": val_mean_answer_reward,
            }
            history["val"].append(val_log)

            # Log a few example rollouts for qualitative inspection.
            gen_logs = log_generations(
                prompts=val_prompts_subset[:3],
                responses=val_responses[:3],
                ground_truths=val_gts[:3],
                reward_infos=val_reward_infos[:3],
            )
            print(
                f"  [val step {step:3d}] "
                f"reward={val_mean_reward:.4f} "
                f"answer_reward={val_mean_answer_reward:.4f}"
            )
            for rec in gen_logs:
                print(f"    GT={rec['ground_truth']} | reward={rec['reward']} | resp={rec['response'][:80]!r}")

    return history
