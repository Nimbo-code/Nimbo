# Copyright (c) 2025, Nimbo
# Licensed under the Apache License, Version 2.0

"""
CoreML Export Module for Apple Neural Engine (ANE) deployment.

This module provides tools to convert HuggingFace LLaMA models to CoreML format
optimized for Apple Neural Engine execution on iOS and macOS devices.

Features:
- Direct HuggingFace to CoreML conversion
- LUT (Look-Up Table) quantization: 4-bit, 6-bit, 8-bit
- Model splitting: embeddings, Decoder, lm_head separately or monolithic
- KV cache management for efficient inference
- Prefill and inference mode support
- Weight deduplication: 15-40% model size reduction for multi-function models
- ANE compatibility checker: Analyze model layers for ANE optimization

Quick Start:
    from nimbo.export.coreml import convert_hf_to_coreml

    # Convert HuggingFace model directly to CoreML
    result = convert_hf_to_coreml(
        model_id="meta-llama/Llama-3.2-1B",
        output_dir="./coreml_output",
        lut_bits=4,
    )

Classes:
    LlamaConverter: Main converter for LLaMA models to CoreML
    LlamaConfig: Configuration for ANE-optimized LLaMA model
    LlamaForCausalLM: ANE-optimized LLaMA model implementation
    BaseConverter: Abstract base class for all converters

HuggingFace Conversion:
    convert_hf_to_coreml: Convert HuggingFace model to CoreML in one call
    convert_hf_to_nimbo: Convert HuggingFace to Nimbo format (intermediate step)
    load_hf_model: Load HuggingFace model for manual conversion

Deduplication:
    combine_models_with_dedup: Combine models with weight sharing (15-40% smaller)
    verify_dedup_correctness: Verify deduplication preserves model outputs
    verify_weight_equivalence: Compare weights between two models

ANE Checker:
    ANEChecker: Analyze model for ANE compatibility
    check_ane_compatibility: Quick function to check and report ANE compatibility
"""

from .base_converter import BaseConverter
from .llama_converter import LlamaConverter
from .llama_model import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRMSNorm,
    LlamaAttention,
    LlamaMLP,
    LlamaDecoderLayer,
    MODEL_DTYPE,
    TEST_DEVICE,
    CONTEXT_LENGTH,
    STATE_LENGTH,
)
from .metadata import AddMetadata, ReadMetadata, get_nimbo_version, ModelPart
from .combine import (
    combine_models,
    combine_models_with_dedup,
    combine_monolithic,
    combine_decoder_chunks,
    validate_chunk_files,
    get_chunk_file_names,
)
from .dedup_weights import (
    prepare_dedup_sources,
    find_replaceable_weights,
    verify_dedup_correctness,
    verify_weight_equivalence,
    ReplacementReason,
    ReplacementDiag,
    DeduplicationReport,
)
from .ane_checker import (
    ANEChecker,
    ANEReport,
    ANELayerReport,
    ANEStatus,
    ANEIssue,
    ANEIssueLevel,
    check_ane_compatibility,
)
from .hf_converter import (
    convert_hf_to_coreml,
    convert_hf_to_nimbo,
    load_hf_model,
    download_hf_model,
    ConversionConfig,
    ConversionResult,
)

__all__ = [
    # Converters
    "BaseConverter",
    "LlamaConverter",
    # Models
    "LlamaConfig",
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaRMSNorm",
    "LlamaAttention",
    "LlamaMLP",
    "LlamaDecoderLayer",
    # Metadata
    "AddMetadata",
    "ReadMetadata",
    "get_nimbo_version",
    "ModelPart",
    # Constants
    "MODEL_DTYPE",
    "TEST_DEVICE",
    "CONTEXT_LENGTH",
    "STATE_LENGTH",
    # Combine utilities
    "combine_models",
    "combine_models_with_dedup",
    "combine_monolithic",
    "combine_decoder_chunks",
    "validate_chunk_files",
    "get_chunk_file_names",
    # Deduplication
    "prepare_dedup_sources",
    "find_replaceable_weights",
    "verify_dedup_correctness",
    "verify_weight_equivalence",
    "ReplacementReason",
    "ReplacementDiag",
    "DeduplicationReport",
    # ANE Checker
    "ANEChecker",
    "ANEReport",
    "ANELayerReport",
    "ANEStatus",
    "ANEIssue",
    "ANEIssueLevel",
    "check_ane_compatibility",
    # HuggingFace Converter
    "convert_hf_to_coreml",
    "convert_hf_to_nimbo",
    "load_hf_model",
    "download_hf_model",
    "ConversionConfig",
    "ConversionResult",
]
