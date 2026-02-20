#!/usr/bin/env python3
"""Benchmark training speed with and without Triton kernel optimizations.

Usage:
    CUDA_VISIBLE_DEVICES=1 python benchmarks/benchmark_training.py
"""

import os
import sys
import time
import gc
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    model_path: str
    batch_size: int = 1
    seq_length: int = 512
    num_steps: int = 20
    warmup_steps: int = 5
    gradient_accumulation_steps: int = 4
    use_bf16: bool = True
    use_gradient_checkpointing: bool = True


def create_dummy_dataset(tokenizer, num_samples: int = 100, seq_length: int = 512) -> Dataset:
    """Create a dummy dataset for benchmarking."""
    # Generate random token sequences
    vocab_size = tokenizer.vocab_size

    data = []
    for _ in range(num_samples):
        # Random tokens (avoiding special tokens)
        input_ids = torch.randint(100, vocab_size - 100, (seq_length,)).tolist()
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        data.append({"text": text})

    return Dataset.from_list(data)


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_model(
    model_path: str,
    use_bf16: bool = True,
    use_gradient_checkpointing: bool = True,
    use_flash_attention: bool = False,  # Disabled by default
) -> tuple:
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}...")

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "cuda",
    }

    if use_bf16:
        model_kwargs["dtype"] = torch.bfloat16

    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.config.use_cache = False
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def benchmark_forward_backward(
    model,
    tokenizer,
    config: BenchmarkConfig,
    label: str = "Benchmark",
) -> Dict[str, float]:
    """Benchmark forward and backward pass."""
    print(f"\n{'='*60}")
    print(f" {label}")
    print(f"{'='*60}")

    device = next(model.parameters()).device

    # Create dummy batch
    input_ids = torch.randint(
        100, tokenizer.vocab_size - 100,
        (config.batch_size, config.seq_length),
        device=device
    )
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    # Warmup
    print(f"Warming up ({config.warmup_steps} steps)...")
    for _ in range(config.warmup_steps):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / config.gradient_accumulation_steps
        loss.backward()
        model.zero_grad()

    torch.cuda.synchronize()
    clear_memory()

    # Benchmark
    print(f"Benchmarking ({config.num_steps} steps)...")

    forward_times = []
    backward_times = []
    total_times = []

    for step in range(config.num_steps):
        torch.cuda.synchronize()

        # Forward pass
        start_forward = time.perf_counter()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        torch.cuda.synchronize()
        end_forward = time.perf_counter()

        # Backward pass
        loss = outputs.loss / config.gradient_accumulation_steps
        start_backward = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        end_backward = time.perf_counter()

        model.zero_grad()

        forward_time = (end_forward - start_forward) * 1000
        backward_time = (end_backward - start_backward) * 1000
        total_time = forward_time + backward_time

        forward_times.append(forward_time)
        backward_times.append(backward_time)
        total_times.append(total_time)

        if (step + 1) % 5 == 0:
            print(f"  Step {step + 1}: {total_time:.2f} ms")

    # Calculate statistics
    results = {
        "forward_avg_ms": sum(forward_times) / len(forward_times),
        "backward_avg_ms": sum(backward_times) / len(backward_times),
        "total_avg_ms": sum(total_times) / len(total_times),
        "forward_min_ms": min(forward_times),
        "backward_min_ms": min(backward_times),
        "total_min_ms": min(total_times),
        "throughput_samples_per_sec": config.batch_size * 1000 / (sum(total_times) / len(total_times)),
        "throughput_tokens_per_sec": config.batch_size * config.seq_length * 1000 / (sum(total_times) / len(total_times)),
    }

    print(f"\nResults:")
    print(f"  Forward:  {results['forward_avg_ms']:.2f} ms (avg), {results['forward_min_ms']:.2f} ms (min)")
    print(f"  Backward: {results['backward_avg_ms']:.2f} ms (avg), {results['backward_min_ms']:.2f} ms (min)")
    print(f"  Total:    {results['total_avg_ms']:.2f} ms (avg), {results['total_min_ms']:.2f} ms (min)")
    print(f"  Throughput: {results['throughput_samples_per_sec']:.2f} samples/s, {results['throughput_tokens_per_sec']:.0f} tokens/s")

    return results


def apply_triton_patches(model) -> Optional[Any]:
    """Apply Triton kernel patches to model."""
    try:
        from nimbo.kernels import is_triton_available, patch_model, get_supported_models

        if not is_triton_available():
            print("Triton not available!")
            return None

        print(f"\nSupported models: {get_supported_models()}")

        stats = patch_model(
            model,
            rms_norm=True,
            swiglu=True,
            rope=True,
            attention=False,
        )

        print(f"\nPatch statistics:\n{stats}")
        return stats

    except Exception as e:
        print(f"Failed to apply patches: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    # Configuration - without gradient checkpointing for max speedup
    config = BenchmarkConfig(
        model_path="/home/elicer/jyp/EXAONE-4.0-1.2B",
        batch_size=4,  # Smaller batch without grad checkpoint
        seq_length=1024,
        num_steps=20,
        warmup_steps=5,
        gradient_accumulation_steps=4,
        use_bf16=True,
        use_gradient_checkpointing=False,  # Disabled for max speedup
    )

    print("="*60)
    print(" EXAONE 4.0 Training Speed Benchmark")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_path}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Sequence length: {config.seq_length}")
    print(f"  Num steps: {config.num_steps}")
    print(f"  BF16: {config.use_bf16}")
    print(f"  Gradient checkpointing: {config.use_gradient_checkpointing}")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ============================================================
    # Test 1: Without Triton kernels
    # ============================================================
    print("\n" + "="*60)
    print(" TEST 1: WITHOUT Triton Kernels")
    print("="*60)

    model, tokenizer = load_model(
        config.model_path,
        use_bf16=config.use_bf16,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
    )

    results_baseline = benchmark_forward_backward(
        model, tokenizer, config,
        label="Baseline (No Triton)"
    )

    # Clean up
    del model
    clear_memory()

    # ============================================================
    # Test 2: With Triton kernels
    # ============================================================
    print("\n" + "="*60)
    print(" TEST 2: WITH Triton Kernels")
    print("="*60)

    model, tokenizer = load_model(
        config.model_path,
        use_bf16=config.use_bf16,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
    )

    # Apply Triton patches
    patch_stats = apply_triton_patches(model)

    if patch_stats is not None:
        results_triton = benchmark_forward_backward(
            model, tokenizer, config,
            label="With Triton Kernels"
        )
    else:
        print("Skipping Triton benchmark due to patch failure")
        results_triton = None

    # Clean up
    del model
    clear_memory()

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)

    print(f"\n{'Metric':<30} {'Baseline':<15} {'Triton':<15} {'Speedup':<10}")
    print("-"*70)

    if results_triton:
        for key in ["forward_avg_ms", "backward_avg_ms", "total_avg_ms"]:
            baseline_val = results_baseline[key]
            triton_val = results_triton[key]
            speedup = baseline_val / triton_val
            print(f"{key:<30} {baseline_val:<15.2f} {triton_val:<15.2f} {speedup:<10.2f}x")

        print("-"*70)

        baseline_throughput = results_baseline["throughput_tokens_per_sec"]
        triton_throughput = results_triton["throughput_tokens_per_sec"]
        throughput_speedup = triton_throughput / baseline_throughput

        print(f"{'Throughput (tokens/sec)':<30} {baseline_throughput:<15.0f} {triton_throughput:<15.0f} {throughput_speedup:<10.2f}x")

        print("\n" + "="*60)
        print(f" Overall Training Speedup: {throughput_speedup:.2f}x")
        print("="*60)
    else:
        print("Could not compare - Triton benchmark failed")


if __name__ == "__main__":
    main()
