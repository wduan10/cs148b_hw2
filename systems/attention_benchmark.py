from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import torch
import time

import torch.nn.functional as F

@dataclass(frozen=True)
class AttentionBenchmarkConfig:
    head_dims: tuple[int, ...] = (16, 32, 64, 128)
    sequence_lengths: tuple[int, ...] = (64, 128, 256, 512, 1024)
    batch_size: int = 8
    forward_passes: int = 100
    backward_passes: int = 100
    compile_attention: bool = False


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark attention implementations.")
    parser.add_argument("--compile-attention", action="store_true")
    return parser


def iter_benchmark_shapes(config: AttentionBenchmarkConfig) -> Iterable[tuple[int, int]]:
    for head_dim in config.head_dims:
        for sequence_length in config.sequence_lengths:
            yield head_dim, sequence_length


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def make_qkv(
    batch_size: int,
    sequence_length: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    q = torch.randn(batch_size, sequence_length, head_dim, device=device, requires_grad=True)
    k = torch.randn(batch_size, sequence_length, head_dim, device=device, requires_grad=True)
    v = torch.randn(batch_size, sequence_length, head_dim, device=device, requires_grad=True)
    return q, k, v


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v)
    return out


def benchmark_attention_once(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> dict[str, float]:
    device = q.device

    # Warmup
    for _ in range(10):
        out = attention(q, k, v)
        loss = out.sum()
        loss.backward()
        q.grad = k.grad = v.grad = None
        sync(device)

    # Forward timing
    sync(device)
    start = time.perf_counter()
    for _ in range(100):
        out = attention(q, k, v)
        sync(device)
    forward_time = time.perf_counter() - start

    # Measure memory before backward
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        out = attention(q, k, v)
        loss = out.sum()
        sync(device)
        memory_before_backward_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    else:
        out = attention(q, k, v)
        loss = out.sum()
        memory_before_backward_mb = float("nan")

    # Backward timing
    sync(device)
    start = time.perf_counter()
    for _ in range(100):
        q.grad = k.grad = v.grad = None
        out = attention(q, k, v)
        loss = out.sum()
        loss.backward()
        sync(device)
    backward_time = time.perf_counter() - start

    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if device.type == "cuda"
        else float("nan")
    )

    return {
        "forward_time_s": forward_time,
        "backward_time_s": backward_time,
        "memory_before_backward_mb": memory_before_backward_mb,
        "peak_memory_mb": peak_memory_mb,
    }


def benchmark_attention_grid(config: AttentionBenchmarkConfig) -> list[dict[str, float | int | str]]:
    device = get_device()
    results = []

    for head_dim, sequence_length in iter_benchmark_shapes(config):
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            q, k, v = make_qkv(
                config.batch_size,
                sequence_length,
                head_dim,
                device,
            )

            result = benchmark_attention_once(q, k, v)

            row = {
                "head_dim": head_dim,
                "sequence_length": sequence_length,
                "status": "ok",
                **result,
            }

        except torch.cuda.OutOfMemoryError:
            row = {
                "head_dim": head_dim,
                "sequence_length": sequence_length,
                "status": "OOM",
                "forward_time_s": float("nan"),
                "backward_time_s": float("nan"),
                "memory_before_backward_mb": float("nan"),
                "peak_memory_mb": float("nan"),
            }
            torch.cuda.empty_cache()

        results.append(row)
        print(row)

    return results


def main() -> None:
    args = build_argparser().parse_args()
    config = AttentionBenchmarkConfig(compile_attention=args.compile_attention)
    benchmark_attention_grid(config)


if __name__ == "__main__":
    main()
