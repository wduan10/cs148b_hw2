from __future__ import annotations

import json
from importlib import import_module
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .prompts import COT_PROMPT_TEMPLATE, DIRECT_PROMPT_TEMPLATE
from .rewards import answer_tag_reward_fn


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_VALIDATION_SIZE = 256


def load_gsm8k_examples(split: str) -> list[dict[str, Any]]:
    """Load GSM8K examples from HuggingFace datasets."""
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split=split)
    examples: list[dict[str, Any]] = []
    for example in dataset:
        answer = str(example["answer"])
        examples.append(
            {
                "question": example["question"],
                "answer": answer,
                "ground_truth": _extract_gsm8k_final_answer(answer),
            }
        )
    return examples


def build_prompts(examples: Sequence[dict[str, Any]], prompt_template: str) -> list[str]:
    """Format raw GSM8K examples into prompt strings."""
    return [prompt_template.format(question=example["question"]) for example in examples]


def evaluate_vllm(
    vllm_model,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: Sequence[str],
    eval_sampling_params,
    ground_truths: Sequence[str] | None = None,
    examples: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate model outputs, score them, and return serializable evaluation artifacts."""
    if ground_truths is None:
        if examples is None:
            raise ValueError("Pass ground_truths or examples so generated responses can be scored.")
        ground_truths = [_example_ground_truth(example) for example in examples]

    if len(prompts) != len(ground_truths):
        raise ValueError(f"Expected one ground truth per prompt, got {len(prompts)} prompts and {len(ground_truths)} labels.")

    request_outputs = vllm_model.generate(list(prompts), eval_sampling_params)
    generations = [_first_vllm_text(output) for output in request_outputs]
    reward_infos = [reward_fn(response, ground_truth) for response, ground_truth in zip(generations, ground_truths, strict=True)]

    records = []
    for idx, (prompt, response, ground_truth, reward_info) in enumerate(
        zip(prompts, generations, ground_truths, reward_infos, strict=True)
    ):
        record: dict[str, Any] = {
            "idx": idx,
            "prompt": prompt,
            "response": response,
            "ground_truth": ground_truth,
            "reward": reward_info,
        }
        if examples is not None:
            record["example"] = dict(examples[idx])
        records.append(record)

    return {
        "metrics": _summarize_rewards(reward_infos),
        "examples": records,
    }


def write_evaluation_results(results: dict[str, Any], output_path: Path) -> None:
    """Serialize generations and scores for later analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def run_direct_baseline(output_path: Path) -> None:
    """Evaluate the direct-prediction GSM8K baseline from Section 3.1."""
    vllm = import_module("vllm")

    examples = load_gsm8k_examples("test")
    prompts = build_prompts(examples, DIRECT_PROMPT_TEMPLATE)
    sampling_params = vllm.SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    model = vllm.LLM(model=DEFAULT_MODEL_NAME)
    results = evaluate_vllm(
        model,
        answer_tag_reward_fn,
        prompts,
        sampling_params,
        examples=examples,
    )
    write_evaluation_results(results, output_path)


def run_cot_baseline(output_path: Path) -> None:
    """Evaluate the chain-of-thought baseline from Section 3.2."""
    raise NotImplementedError


def run_self_consistency_baseline(output_path: Path, k: int = 5) -> None:
    """Evaluate the self-consistency baseline from Section 3.2."""
    raise NotImplementedError


def get_prompt_template(use_cot: bool) -> str:
    return COT_PROMPT_TEMPLATE if use_cot else DIRECT_PROMPT_TEMPLATE


def _extract_gsm8k_final_answer(answer: str) -> str:
    """Return the final GSM8K answer after the #### delimiter."""
    return answer.rsplit("####", maxsplit=1)[-1].strip()


def _example_ground_truth(example: dict[str, Any]) -> str:
    if "ground_truth" in example:
        return str(example["ground_truth"])
    return _extract_gsm8k_final_answer(str(example["answer"]))


def _first_vllm_text(output: Any) -> str:
    return output.outputs[0].text


def _summarize_rewards(reward_infos: Sequence[dict[str, float]]) -> dict[str, Any]:
    n_examples = len(reward_infos)
    totals = {
        "reward": 0.0,
        "format_reward": 0.0,
        "answer_reward": 0.0,
    }
    category_counts = {
        "correct_format_and_answer": 0,
        "formatted_wrong_answer": 0,
        "unformatted": 0,
    }

    for reward_info in reward_infos:
        for key in totals:
            totals[key] += float(reward_info.get(key, 0.0))

        format_reward = float(reward_info.get("format_reward", 0.0))
        answer_reward = float(reward_info.get("answer_reward", 0.0))
        if format_reward == 1.0 and answer_reward == 1.0:
            category_counts["correct_format_and_answer"] += 1
        elif format_reward == 1.0 and answer_reward == 0.0:
            category_counts["formatted_wrong_answer"] += 1
        elif format_reward == 0.0 and answer_reward == 0.0:
            category_counts["unformatted"] += 1

    if n_examples == 0:
        return {
            "n_examples": 0,
            "mean_reward": 0.0,
            "mean_format_reward": 0.0,
            "mean_answer_reward": 0.0,
            "category_counts": category_counts,
        }

    return {
        "n_examples": n_examples,
        "mean_reward": totals["reward"] / n_examples,
        "mean_format_reward": totals["format_reward"] / n_examples,
        "mean_answer_reward": totals["answer_reward"] / n_examples,
        "category_counts": category_counts,
    }
