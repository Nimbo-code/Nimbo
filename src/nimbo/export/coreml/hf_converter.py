# Copyright (c) 2025, Nimbo
# Licensed under MIT License

"""
HuggingFace to CoreML Converter

This module provides a simple API to convert HuggingFace LLaMA models
directly to CoreML format optimized for Apple Neural Engine.

Usage:
    from nimbo.export.coreml import convert_hf_to_coreml

    # Simple conversion
    convert_hf_to_coreml(
        model_id="meta-llama/Llama-3.2-1B",
        output_dir="./coreml_output",
        lut_bits=4,
    )

    # With options
    convert_hf_to_coreml(
        model_id="meta-llama/Llama-3.2-3B",
        output_dir="./output",
        lut_bits=4,
        context_length=512,
        split_model=True,
        num_chunks=4,
    )
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass

import torch
import torch.nn as nn

# Check for optional dependencies
try:
    from transformers import AutoConfig, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False


@dataclass
class ConversionConfig:
    """Configuration for HuggingFace to CoreML conversion."""
    # Model settings
    context_length: int = 512
    state_length: int = 512
    batch_size: int = 64  # For prefill mode

    # Quantization
    lut_bits: int = 4  # 4, 6, or 8

    # Model splitting
    split_model: bool = False
    num_chunks: int = 1

    # Output options
    include_prefill: bool = True
    combine_models: bool = True
    use_dedup: bool = True

    # Advanced
    compute_units: str = "ALL"  # ALL, CPU_AND_NE, CPU_ONLY


@dataclass
class ConversionResult:
    """Result of HuggingFace to CoreML conversion."""
    success: bool
    output_paths: List[str]
    model_name: str
    config: Dict[str, Any]
    errors: List[str]
    warnings: List[str]


def _check_dependencies():
    """Check if required dependencies are available."""
    missing = []
    if not HAS_TRANSFORMERS:
        missing.append("transformers")
    if not HAS_HF_HUB:
        missing.append("huggingface_hub")

    if missing:
        raise ImportError(
            f"Missing required packages: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )


def _map_hf_config_to_llama_config(hf_config) -> Dict[str, Any]:
    """Map HuggingFace config to Nimbo LlamaConfig format."""
    config_dict = {
        "hidden_size": hf_config.hidden_size,
        "vocab_size": hf_config.vocab_size,
        "num_hidden_layers": hf_config.num_hidden_layers,
        "num_attention_heads": hf_config.num_attention_heads,
        "num_key_value_heads": getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
        "intermediate_size": hf_config.intermediate_size,
        "hidden_act": getattr(hf_config, "hidden_act", "silu"),
        "rms_norm_eps": getattr(hf_config, "rms_norm_eps", 1e-6),
        "rope_theta": getattr(hf_config, "rope_theta", 10000.0),
        "max_position_embeddings": getattr(hf_config, "max_position_embeddings", 4096),
        "tie_word_embeddings": getattr(hf_config, "tie_word_embeddings", False),
    }

    # Handle head_dim
    if hasattr(hf_config, "head_dim"):
        config_dict["head_dim"] = hf_config.head_dim
    else:
        config_dict["head_dim"] = config_dict["hidden_size"] // config_dict["num_attention_heads"]

    return config_dict


def _extract_weights_from_hf_model(hf_model, output_dir: str):
    """Extract and save weights from HuggingFace model to safetensors format."""
    import safetensors.torch

    os.makedirs(output_dir, exist_ok=True)

    state_dict = hf_model.state_dict()

    # Save all weights to a single safetensors file
    output_path = os.path.join(output_dir, "model.safetensors")
    safetensors.torch.save_file(state_dict, output_path)

    print(f"  Saved {len(state_dict)} tensors to {output_path}")
    return output_path


def download_hf_model(
    model_id: str,
    output_dir: str = None,
    token: str = None,
) -> str:
    """Download HuggingFace model to local directory.

    Args:
        model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3.2-1B")
        output_dir: Output directory (default: ./models/{model_name})
        token: HuggingFace token for gated models

    Returns:
        Path to downloaded model directory
    """
    _check_dependencies()

    if output_dir is None:
        model_name = model_id.split("/")[-1]
        output_dir = f"./models/{model_name}"

    print(f"Downloading {model_id}...")

    try:
        local_dir = snapshot_download(
            model_id,
            local_dir=output_dir,
            token=token,
            ignore_patterns=["*.bin", "*.h5", "*.ot", "*.msgpack"],  # Prefer safetensors
        )
        print(f"Model downloaded to: {local_dir}")
        return local_dir
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}")


def load_hf_model(
    model_id_or_path: str,
    torch_dtype: torch.dtype = torch.float16,
    device: str = "cpu",
    token: str = None,
):
    """Load a HuggingFace model.

    Args:
        model_id_or_path: HuggingFace model ID or local path
        torch_dtype: Torch dtype for model weights
        device: Device to load model on
        token: HuggingFace token for gated models

    Returns:
        Tuple of (model, config)
    """
    _check_dependencies()

    print(f"Loading HuggingFace model: {model_id_or_path}")

    # Load config
    config = AutoConfig.from_pretrained(
        model_id_or_path,
        trust_remote_code=True,
        token=token,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        config=config,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
        token=token,
    )

    print(f"  Model loaded: {config.architectures[0] if config.architectures else 'Unknown'}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, config


def convert_hf_to_nimbo(
    model_id_or_path: str,
    output_dir: str,
    context_length: int = 512,
    state_length: int = 512,
    token: str = None,
) -> str:
    """Convert HuggingFace model to Nimbo's ANE-optimized format.

    Args:
        model_id_or_path: HuggingFace model ID or local path
        output_dir: Output directory for converted model
        context_length: Maximum context length
        state_length: KV cache state length
        token: HuggingFace token for gated models

    Returns:
        Path to converted model directory
    """
    from .llama_model import LlamaConfig, LlamaForCausalLM

    _check_dependencies()

    os.makedirs(output_dir, exist_ok=True)

    # Load HuggingFace model
    hf_model, hf_config = load_hf_model(
        model_id_or_path,
        torch_dtype=torch.float16,
        device="cpu",
        token=token,
    )

    # Map config
    print("Converting model configuration...")
    config_dict = _map_hf_config_to_llama_config(hf_config)
    config_dict["context_length"] = context_length
    config_dict["state_length"] = state_length

    # Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"  Config saved to: {config_path}")

    # Extract and save weights
    print("Extracting model weights...")
    _extract_weights_from_hf_model(hf_model, output_dir)

    # Create Nimbo model and load weights
    print("Creating Nimbo ANE-optimized model...")
    nimbo_config = LlamaConfig(**config_dict)
    nimbo_model = LlamaForCausalLM(nimbo_config)
    nimbo_model.load_pretrained_weights(output_dir)

    # Clean up HF model to save memory
    del hf_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"Nimbo model ready at: {output_dir}")
    return output_dir, nimbo_model, nimbo_config


def convert_hf_to_coreml(
    model_id_or_path: str,
    output_dir: str,
    lut_bits: int = 4,
    context_length: int = 512,
    state_length: int = 512,
    batch_size: int = 64,
    split_model: bool = False,
    num_chunks: int = 1,
    include_prefill: bool = True,
    combine_models: bool = True,
    use_dedup: bool = True,
    token: str = None,
    config: ConversionConfig = None,
) -> ConversionResult:
    """Convert HuggingFace model directly to CoreML format.

    This is the main entry point for HF to CoreML conversion.

    Args:
        model_id_or_path: HuggingFace model ID (e.g., "meta-llama/Llama-3.2-1B") or local path
        output_dir: Directory to save CoreML models
        lut_bits: LUT quantization bits (4, 6, or 8)
        context_length: Maximum sequence length
        state_length: KV cache state length
        batch_size: Batch size for prefill mode
        split_model: If True, split model into chunks
        num_chunks: Number of chunks (if split_model=True)
        include_prefill: If True, also convert prefill mode
        combine_models: If True, combine infer+prefill into multi-function model
        use_dedup: If True, use weight deduplication when combining
        token: HuggingFace token for gated models
        config: ConversionConfig object (overrides individual parameters)

    Returns:
        ConversionResult with output paths and status
    """
    try:
        import coremltools as ct
    except ImportError:
        return ConversionResult(
            success=False,
            output_paths=[],
            model_name=model_id_or_path,
            config={},
            errors=["coremltools not available. Install with: pip install coremltools"],
            warnings=[],
        )

    from .llama_converter import LlamaConverter
    from .combine import combine_monolithic

    # Use config object if provided
    if config is not None:
        lut_bits = config.lut_bits
        context_length = config.context_length
        state_length = config.state_length
        batch_size = config.batch_size
        split_model = config.split_model
        num_chunks = config.num_chunks
        include_prefill = config.include_prefill
        combine_models = config.combine_models
        use_dedup = config.use_dedup

    errors = []
    warnings = []
    output_paths = []

    # Get model name
    model_name = model_id_or_path.split("/")[-1] if "/" in model_id_or_path else os.path.basename(model_id_or_path)

    print("=" * 60)
    print(f"Converting {model_name} to CoreML")
    print("=" * 60)
    print(f"  LUT bits: {lut_bits}")
    print(f"  Context length: {context_length}")
    print(f"  Split model: {split_model} (chunks: {num_chunks})")
    print(f"  Include prefill: {include_prefill}")
    print(f"  Combine models: {combine_models}")
    print()

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, ".temp")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Step 1: Convert HF to Nimbo format
        print("Step 1: Converting HuggingFace model to Nimbo format...")
        nimbo_dir, nimbo_model, nimbo_config = convert_hf_to_nimbo(
            model_id_or_path,
            temp_dir,
            context_length=context_length,
            state_length=state_length,
            token=token,
        )

        # Step 2: Create CoreML converter
        print("\nStep 2: Initializing CoreML converter...")
        converter = LlamaConverter(
            model=nimbo_model,
            context_length=context_length,
            state_length=state_length,
            lut_bits=lut_bits,
            batch_size=batch_size,
        )

        # Step 3: Convert model(s)
        if split_model and num_chunks > 1:
            print(f"\nStep 3: Converting model in {num_chunks} chunks...")
            converter.num_chunks = num_chunks

            for chunk_idx in range(num_chunks):
                print(f"\n  Chunk {chunk_idx + 1}/{num_chunks}:")

                # Inference mode
                infer_path = os.path.join(
                    output_dir,
                    f"{model_name}_decoder_lut{lut_bits}_chunk{chunk_idx + 1:02d}of{num_chunks:02d}.mlpackage"
                )
                print(f"    Converting inference mode...")
                infer_model = converter.convert(split_part="2", chunk_idx=chunk_idx)
                infer_model.save(infer_path)
                output_paths.append(infer_path)
                print(f"    Saved: {infer_path}")

                # Prefill mode
                if include_prefill:
                    prefill_path = os.path.join(
                        output_dir,
                        f"{model_name}_decoder_prefill_lut{lut_bits}_chunk{chunk_idx + 1:02d}of{num_chunks:02d}.mlpackage"
                    )
                    print(f"    Converting prefill mode...")
                    prefill_model = converter.convert(split_part="2_prefill", chunk_idx=chunk_idx)
                    prefill_model.save(prefill_path)
                    output_paths.append(prefill_path)
                    print(f"    Saved: {prefill_path}")

                    # Combine if requested
                    if combine_models:
                        combined_path = os.path.join(
                            output_dir,
                            f"{model_name}_decoder_full_lut{lut_bits}_chunk{chunk_idx + 1:02d}of{num_chunks:02d}.mlpackage"
                        )
                        print(f"    Combining models...")
                        combine_monolithic(
                            infer_path, prefill_path, combined_path,
                            use_dedup=use_dedup
                        )
                        output_paths.append(combined_path)

                        # Remove individual files if combined
                        os.remove(infer_path) if os.path.exists(infer_path) else None
                        os.remove(prefill_path) if os.path.exists(prefill_path) else None
                        output_paths = [p for p in output_paths if p != infer_path and p != prefill_path]

            # Convert embeddings and lm_head separately
            print("\n  Converting embeddings...")
            embed_path = os.path.join(output_dir, f"{model_name}_embeddings.mlpackage")
            embed_model = converter.convert(split_part="1")
            embed_model.save(embed_path)
            output_paths.append(embed_path)

            print("  Converting lm_head...")
            lmhead_path = os.path.join(output_dir, f"{model_name}_lm_head_lut{lut_bits}.mlpackage")
            lmhead_model = converter.convert(split_part="3")
            lmhead_model.save(lmhead_path)
            output_paths.append(lmhead_path)

        else:
            # Monolithic conversion
            print("\nStep 3: Converting monolithic model...")

            # Inference mode
            infer_path = os.path.join(output_dir, f"{model_name}_monolithic_lut{lut_bits}.mlpackage")
            print("  Converting inference mode...")
            infer_model = converter.convert(split_part="monolithic")
            infer_model.save(infer_path)
            output_paths.append(infer_path)
            print(f"  Saved: {infer_path}")

            # Prefill mode
            if include_prefill:
                prefill_path = os.path.join(output_dir, f"{model_name}_monolithic_prefill_lut{lut_bits}.mlpackage")
                print("  Converting prefill mode...")
                prefill_model = converter.convert(split_part="monolithic_prefill")
                prefill_model.save(prefill_path)
                output_paths.append(prefill_path)
                print(f"  Saved: {prefill_path}")

                # Combine
                if combine_models:
                    combined_path = os.path.join(output_dir, f"{model_name}_full_lut{lut_bits}.mlpackage")
                    print("  Combining inference + prefill...")
                    result = combine_monolithic(
                        infer_path, prefill_path, combined_path,
                        use_dedup=use_dedup
                    )
                    output_paths.append(combined_path)

                    # Clean up individual files
                    shutil.rmtree(infer_path, ignore_errors=True)
                    shutil.rmtree(prefill_path, ignore_errors=True)
                    output_paths = [p for p in output_paths if p != infer_path and p != prefill_path]

        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        print("\n" + "=" * 60)
        print("Conversion Complete!")
        print("=" * 60)
        print("\nOutput files:")
        for path in output_paths:
            if os.path.exists(path):
                size_mb = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, _, filenames in os.walk(path)
                    for filename in filenames
                ) / (1024 * 1024)
                print(f"  {os.path.basename(path)}: {size_mb:.1f} MB")

        return ConversionResult(
            success=True,
            output_paths=output_paths,
            model_name=model_name,
            config={
                "lut_bits": lut_bits,
                "context_length": context_length,
                "num_chunks": num_chunks,
            },
            errors=errors,
            warnings=warnings,
        )

    except Exception as e:
        import traceback
        errors.append(f"Conversion failed: {str(e)}")
        errors.append(traceback.format_exc())

        # Cleanup on failure
        shutil.rmtree(temp_dir, ignore_errors=True)

        return ConversionResult(
            success=False,
            output_paths=[],
            model_name=model_name,
            config={},
            errors=errors,
            warnings=warnings,
        )


# Convenience aliases
hf_to_coreml = convert_hf_to_coreml
from_huggingface = convert_hf_to_coreml
