#!/usr/bin/env python3
"""Benchmark Nimbo Triton kernels against PyTorch implementations.

Usage:
    python benchmarks/benchmark_kernels.py

Measures:
    - RMSNorm: Forward and backward pass speedup
    - SwiGLU: Forward and backward pass speedup
    - RoPE: Forward pass speedup for Q+K rotation
"""

import torch
import torch.nn.functional as F
import time
import argparse
from typing import Callable, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    pytorch_time_ms: float
    triton_time_ms: float
    speedup: float
    config: Dict


def benchmark_function(
    fn: Callable,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
) -> float:
    """Benchmark a function and return average time in milliseconds."""
    torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup_iters):
        fn()

    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(benchmark_iters):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / benchmark_iters * 1000


def pytorch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Reference PyTorch RMSNorm."""
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return (x_norm * weight).to(x.dtype)


def pytorch_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Reference PyTorch SwiGLU."""
    return F.silu(gate) * up


def benchmark_rms_norm(
    batch_sizes: List[int],
    seq_lengths: List[int],
    hidden_sizes: List[int],
    dtype: torch.dtype = torch.float16,
) -> List[BenchmarkResult]:
    """Benchmark RMSNorm kernel."""
    from nimbo.kernels import triton_rms_norm

    results = []
    device = torch.device("cuda")

    for batch in batch_sizes:
        for seq in seq_lengths:
            for hidden in hidden_sizes:
                config = {"batch": batch, "seq": seq, "hidden": hidden, "dtype": str(dtype)}

                x = torch.randn(batch, seq, hidden, device=device, dtype=dtype)
                weight = torch.ones(hidden, device=device, dtype=dtype)

                # PyTorch
                pytorch_time = benchmark_function(
                    lambda: pytorch_rms_norm(x, weight)
                )

                # Triton
                triton_time = benchmark_function(
                    lambda: triton_rms_norm(x, weight)
                )

                speedup = pytorch_time / triton_time

                results.append(BenchmarkResult(
                    name="RMSNorm Forward",
                    pytorch_time_ms=pytorch_time,
                    triton_time_ms=triton_time,
                    speedup=speedup,
                    config=config,
                ))

    return results


def benchmark_swiglu(
    batch_seq_sizes: List[int],
    intermediate_sizes: List[int],
    dtype: torch.dtype = torch.float16,
) -> List[BenchmarkResult]:
    """Benchmark SwiGLU kernel."""
    from nimbo.kernels import triton_swiglu_forward

    results = []
    device = torch.device("cuda")

    for batch_seq in batch_seq_sizes:
        for intermediate in intermediate_sizes:
            config = {"batch_seq": batch_seq, "intermediate": intermediate, "dtype": str(dtype)}

            gate = torch.randn(batch_seq, intermediate, device=device, dtype=dtype)
            up = torch.randn(batch_seq, intermediate, device=device, dtype=dtype)

            # PyTorch
            pytorch_time = benchmark_function(
                lambda: pytorch_swiglu(gate, up)
            )

            # Triton
            triton_time = benchmark_function(
                lambda: triton_swiglu_forward(gate.contiguous(), up.contiguous())
            )

            speedup = pytorch_time / triton_time

            results.append(BenchmarkResult(
                name="SwiGLU Forward",
                pytorch_time_ms=pytorch_time,
                triton_time_ms=triton_time,
                speedup=speedup,
                config=config,
            ))

    return results


def benchmark_rope(
    batch_sizes: List[int],
    seq_lengths: List[int],
    num_heads_list: List[int],
    head_dim: int = 128,
    dtype: torch.dtype = torch.float32,
) -> List[BenchmarkResult]:
    """Benchmark RoPE kernel."""
    from nimbo.kernels import NimboRoPE
    from nimbo.kernels.triton.rope import compute_rope_cache

    results = []
    device = torch.device("cuda")

    def pytorch_rope(q, k, cos, sin):
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        cos_exp = cos.unsqueeze(0).unsqueeze(2)
        sin_exp = sin.unsqueeze(0).unsqueeze(2)
        cos_full = torch.cat([cos_exp, cos_exp], dim=-1)
        sin_full = torch.cat([sin_exp, sin_exp], dim=-1)

        q_embed = (q * cos_full) + (rotate_half(q) * sin_full)
        k_embed = (k * cos_full) + (rotate_half(k) * sin_full)
        return q_embed, k_embed

    for batch in batch_sizes:
        for seq in seq_lengths:
            for num_heads in num_heads_list:
                config = {
                    "batch": batch,
                    "seq": seq,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "dtype": str(dtype),
                }

                q = torch.randn(batch, seq, num_heads, head_dim, device=device, dtype=dtype)
                k = torch.randn(batch, seq, num_heads, head_dim, device=device, dtype=dtype)

                # Compute cache
                cos, sin = compute_rope_cache(
                    seq_len=seq,
                    head_dim=head_dim,
                    device=device,
                    dtype=dtype,
                )

                # Triton RoPE module
                rope = NimboRoPE(
                    head_dim=head_dim,
                    max_seq_len=seq,
                    device=device,
                    dtype=dtype,
                    rope_type="default",
                )

                # PyTorch
                pytorch_time = benchmark_function(
                    lambda: pytorch_rope(q, k, cos, sin)
                )

                # Triton
                triton_time = benchmark_function(
                    lambda: rope(q, k)
                )

                speedup = pytorch_time / triton_time

                results.append(BenchmarkResult(
                    name="RoPE Forward",
                    pytorch_time_ms=pytorch_time,
                    triton_time_ms=triton_time,
                    speedup=speedup,
                    config=config,
                ))

    return results


def print_results(results: List[BenchmarkResult], title: str):
    """Print benchmark results in a formatted table."""
    print(f"\n{'=' * 80}")
    print(f" {title}")
    print(f"{'=' * 80}")

    # Group by config
    for result in results:
        config_str = ", ".join(f"{k}={v}" for k, v in result.config.items() if k != "dtype")
        print(f"\n{result.name} ({config_str})")
        print(f"  PyTorch:  {result.pytorch_time_ms:8.3f} ms")
        print(f"  Triton:   {result.triton_time_ms:8.3f} ms")
        print(f"  Speedup:  {result.speedup:8.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Nimbo Triton kernels")
    parser.add_argument("--kernel", type=str, default="all", choices=["all", "rms_norm", "swiglu", "rope"])
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print("\n" + "=" * 80)
    print(" Nimbo Triton Kernel Benchmarks")
    print("=" * 80)
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"Data type: {dtype}")

    # EXAONE 7.8B typical sizes
    # - hidden_size: 4096
    # - intermediate_size: 14336
    # - num_attention_heads: 32
    # - num_key_value_heads: 8
    # - head_dim: 128

    if args.kernel in ["all", "rms_norm"]:
        results = benchmark_rms_norm(
            batch_sizes=[1, 4],
            seq_lengths=[512, 2048],
            hidden_sizes=[2048, 4096],
            dtype=dtype,
        )
        print_results(results, "RMSNorm Benchmark Results")

    if args.kernel in ["all", "swiglu"]:
        results = benchmark_swiglu(
            batch_seq_sizes=[512, 2048, 8192],
            intermediate_sizes=[5504, 11008, 14336],  # LLaMA/EXAONE intermediate sizes
            dtype=dtype,
        )
        print_results(results, "SwiGLU Benchmark Results")

    if args.kernel in ["all", "rope"]:
        results = benchmark_rope(
            batch_sizes=[1, 2],
            seq_lengths=[512, 2048],
            num_heads_list=[32],
            head_dim=128,
            dtype=torch.float32,  # RoPE needs higher precision
        )
        print_results(results, "RoPE Benchmark Results")

    print("\n" + "=" * 80)
    print(" Benchmark Complete")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
