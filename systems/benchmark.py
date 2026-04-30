from __future__ import annotations

import argparse
import timeit
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F

from basics.basics.model import BasicsTransformerLM


@dataclass(frozen=True)
class ModelSpec:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "small": ModelSpec(d_model=512, d_ff=2048, num_layers=8, num_heads=8),
    "medium": ModelSpec(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "large": ModelSpec(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    model_size: str
    context_length: int = 128
    batch_size: int = 4
    vocab_size: int = 10_000
    warmup_steps: int = 5
    measure_steps: int = 10
    mode: Literal["forward", "forward-backward", "train-step"] = "forward"
    use_bf16: bool = False
    use_memory_profiler: bool = False
    compile_model: bool = False
    output_dir: Path = Path("artifacts")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark and profile the Basics transformer.")
    parser.add_argument("--model-size", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--mode", choices=["forward", "forward-backward", "train-step"], default="forward")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--use-memory-profiler", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def build_model(config: BenchmarkConfig) -> torch.nn.Module:
    """Instantiate the staff Basics transformer for the requested model size."""
    spec = MODEL_SPECS[config.model_size]

    model = BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=spec.d_model,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        d_ff=spec.d_ff,
        rope_theta=10_000.0,
    )

    return model


def make_random_batch(config: BenchmarkConfig, device: torch.device) -> torch.Tensor:
    """Construct a random token batch for benchmarking and profiling."""
    return torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.context_length),
        device=device,
        dtype=torch.long,
    )


def run_single_step(
    model: torch.nn.Module,
    batch: torch.Tensor,
    mode: Literal["forward", "forward-backward", "train-step"],
    autocast_context,
) -> None:
    """Execute one benchmark step and synchronize CUDA before returning."""
    device = batch.device

    if mode == "forward":
        with torch.no_grad():
            with autocast_context:
                _ = model(batch)

    elif mode == "forward-backward":
        model.zero_grad(set_to_none=True)

        with autocast_context:
            logits = model(batch)
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1),
            )

        loss.backward()

    elif mode == "train-step":
        if not hasattr(model, "_benchmark_optimizer"):
            model._benchmark_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        optimizer = model._benchmark_optimizer
        optimizer.zero_grad(set_to_none=True)

        with autocast_context:
            logits = model(batch)
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1),
            )

        loss.backward()
        optimizer.step()

    else:
        raise ValueError(f"Unknown mode: {mode}")

    synchronize(device)


def benchmark_model(config: BenchmarkConfig) -> dict[str, float]:
    """Run warmup steps followed by timed measurement steps."""
    device = get_device()
    model = build_model(config).to(device)
    model.train()

    if config.compile_model:
        model = torch.compile(model)

    batch = make_random_batch(config, device)
    autocast_context = make_autocast_context(config.use_bf16)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    maybe_start_memory_history(config.use_memory_profiler)

    for _ in range(config.warmup_steps):
        run_single_step(model, batch, config.mode, autocast_context)

    times: list[float] = []

    for _ in range(config.measure_steps):
        start = timeit.default_timer()
        run_single_step(model, batch, config.mode, autocast_context)
        end = timeit.default_timer()
        times.append(end - start)

    maybe_dump_memory_snapshot(
        config.use_memory_profiler,
        config.output_dir / f"{config.model_size}_{config.mode}_memory.pickle",
    )

    times_tensor = torch.tensor(times)
    avg_time = times_tensor.mean().item()
    std_time = times_tensor.std(unbiased=False).item()

    results = {
        "avg_time_sec": avg_time,
        "std_time_sec": std_time,
    }

    print("Benchmark results")
    print("-----------------")
    print(f"device:        {device}")
    print(f"model size:    {config.model_size}")
    print(f"mode:          {config.mode}")
    print(f"bf16:          {config.use_bf16}")
    print(f"compiled:      {config.compile_model}")
    print(f"batch size:    {config.batch_size}")
    print(f"context len:   {config.context_length}")
    print(f"warmup steps:  {config.warmup_steps}")
    print(f"measure steps: {config.measure_steps}")
    print()
    print(f"avg time:      {avg_time:.6f} s")
    print(f"std time:      {std_time:.6f} s")

    return results


def annotated_scaled_dot_product_attention(*args, **kwargs):
    """Optional NVTX-annotated attention path for Nsight Systems profiling."""
    if torch.cuda.is_available():
        with torch.cuda.nvtx.range("scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(*args, **kwargs)

    return F.scaled_dot_product_attention(*args, **kwargs)


def maybe_start_memory_history(enabled: bool) -> None:
    if enabled:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA memory profiler requires a CUDA device.")

        torch.cuda.memory._record_memory_history(max_entries=100_000)


def maybe_dump_memory_snapshot(enabled: bool, output_path: Path) -> None:
    if enabled:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA memory profiler requires a CUDA device.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(output_path))


def make_autocast_context(use_bf16: bool):
    if use_bf16:
        if not torch.cuda.is_available():
            raise RuntimeError("--use-bf16 is currently configured for CUDA autocast only.")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    return nullcontext()


def main() -> None:
    args = build_argparser().parse_args()
    config = BenchmarkConfig(
        model_size=args.model_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        mode=args.mode,
        use_bf16=args.use_bf16,
        use_memory_profiler=args.use_memory_profiler,
        compile_model=args.compile_model,
        output_dir=args.output_dir,
    )
    benchmark_model(config)


if __name__ == "__main__":
    main()