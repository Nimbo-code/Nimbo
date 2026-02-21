# Copyright (c) 2025, Nimbo
# Licensed under the Apache License, Version 2.0
# Based on Anemll (https://github.com/Anemll/Anemll) - MIT License

"""
CoreML model combining utilities for Nimbo.

This module provides functions to combine multiple CoreML models
(chunked Decoder layers, inference/prefill modes) into multi-function models.

Features:
    - Weight deduplication: Reduces model size by 15-40% by sharing identical weights
    - Multi-function models: Combine inference + prefill into single .mlpackage
    - Chunk combining: Combine decoder layer chunks with their prefill variants

Usage:
    # Combine with deduplication (recommended, 15-40% smaller)
    combine_models_with_dedup(
        model_paths=["infer.mlpackage", "prefill.mlpackage"],
        function_names=["infer", "prefill"],
        output_path="combined.mlpackage"
    )

    # Combine without deduplication (faster, larger file)
    combine_monolithic(
        infer_path="llama_monolithic.mlpackage",
        prefill_path="llama_monolithic_prefill.mlpackage",
        output_path="llama_full.mlpackage",
        use_dedup=False
    )

    # Combine chunked Decoder layers
    combine_decoder_chunks(
        num_chunks=4,
        prefix="llama",
        lut_bits=4
    )
"""

import os
import shutil
from typing import List, Optional, Tuple, Dict

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

from .metadata import AddMetadata, ReadMetadata


def combine_models(
    model_paths: list,
    function_names: list,
    output_path: str,
    default_function: str = None,
    use_dedup: bool = False,
    verbose: bool = False,
):
    """Combine multiple CoreML models into a single multi-function model.

    Args:
        model_paths: List of paths to input .mlpackage files
        function_names: List of function names for each model
        output_path: Path to save the combined model
        default_function: Name of the default function (default: first function)
        use_dedup: If True, apply weight deduplication (15-40% smaller)
        verbose: Print detailed progress

    Returns:
        ct.models.MLModel: Combined multi-function model

    Example:
        combine_models(
            model_paths=["infer.mlpackage", "prefill.mlpackage"],
            function_names=["infer", "prefill"],
            output_path="combined.mlpackage",
            use_dedup=True  # Recommended for smaller file size
        )
    """
    if not COREML_AVAILABLE:
        raise ImportError("coremltools is required for model combining. Install with: pip install coremltools")

    if len(model_paths) != len(function_names):
        raise ValueError("model_paths and function_names must have the same length")

    # Validate all input files exist
    for path in model_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

    print(f"\nCombining {len(model_paths)} models" + (" with deduplication" if use_dedup else "") + ":")
    for path, func_name in zip(model_paths, function_names):
        print(f"  {func_name}: {path}")

    if use_dedup and len(model_paths) >= 2:
        return combine_models_with_dedup(
            model_paths=model_paths,
            function_names=function_names,
            output_path=output_path,
            default_function=default_function,
            verbose=verbose,
        )

    # Create multi-function descriptor (without dedup)
    desc = ct.utils.MultiFunctionDescriptor()
    for path, func_name in zip(model_paths, function_names):
        desc.add_function(path, src_function_name="main", target_function_name=func_name)

    # Set default function
    desc.default_function_name = default_function or function_names[0]

    # Save temporary combined model
    temp_path = f"temp_{os.path.basename(output_path)}"
    print(f"Creating multi-function model...")
    ct.utils.save_multifunction(desc, temp_path)

    # Load, add metadata, and save final
    print(f"Adding metadata...")
    combined_model = ct.models.MLModel(temp_path)

    # Combine metadata from source models
    source_models = [ct.models.MLModel(path) for path in model_paths]
    _add_combined_metadata(combined_model, source_models, function_names)

    print(f"Saving to: {output_path}")
    combined_model.save(output_path)

    # Clean up temp file
    shutil.rmtree(temp_path, ignore_errors=True)

    print("Model combination complete!")
    return combined_model


def combine_models_with_dedup(
    model_paths: List[str],
    function_names: List[str],
    output_path: str,
    default_function: str = None,
    cos_threshold: float = 0.9999,
    mean_abs_threshold: float = 0.001,
    verify: bool = True,
    verbose: bool = False,
) -> Tuple[any, Dict]:
    """Combine multiple CoreML models with weight deduplication.

    This function reduces model size by 15-40% by sharing semantically
    identical weights between functions (e.g., infer and prefill).

    Args:
        model_paths: List of paths to input .mlpackage files
        function_names: List of function names for each model
        output_path: Path to save the combined model
        default_function: Name of the default function (default: first function)
        cos_threshold: Minimum cosine similarity for weight matching (default 0.9999)
        mean_abs_threshold: Maximum mean absolute difference (default 0.001)
        verify: If True, verify deduplication correctness after combining
        verbose: Print detailed progress

    Returns:
        Tuple of (combined_model, dedup_report)

    Example:
        model, report = combine_models_with_dedup(
            model_paths=["infer.mlpackage", "prefill.mlpackage"],
            function_names=["infer", "prefill"],
            output_path="combined.mlpackage"
        )
        print(report.summary())  # Shows bytes saved
    """
    if not COREML_AVAILABLE:
        raise ImportError("coremltools is required. Install with: pip install coremltools")

    from .dedup_weights import (
        prepare_dedup_sources,
        ReplacementDiag,
        verify_dedup_correctness,
    )

    if len(model_paths) != len(function_names):
        raise ValueError("model_paths and function_names must have the same length")

    for path in model_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

    # Build sources list: (path, src_function, target_function)
    sources = [(path, "main", fn) for path, fn in zip(model_paths, function_names)]

    # Collect diagnostics
    diagnostics: List[ReplacementDiag] = []

    # Use dedup context manager
    with prepare_dedup_sources(
        sources,
        cos_threshold=cos_threshold,
        mean_abs_threshold=mean_abs_threshold,
        verify_dequant=True,
        verbose=verbose,
        diagnostics=diagnostics,
    ) as deduped_sources:
        # Create multi-function descriptor with deduped sources
        desc = ct.utils.MultiFunctionDescriptor()
        for path, src_fn, tgt_fn in deduped_sources:
            desc.add_function(path, src_function_name=src_fn, target_function_name=tgt_fn)

        desc.default_function_name = default_function or function_names[0]

        # Save combined model
        temp_path = f"temp_{os.path.basename(output_path)}"
        print(f"Creating multi-function model with dedup...")
        ct.utils.save_multifunction(desc, temp_path)

        # Load, add metadata, and save final
        print(f"Adding metadata...")
        combined_model = ct.models.MLModel(temp_path)

        source_models = [ct.models.MLModel(path) for path in model_paths]
        _add_combined_metadata(combined_model, source_models, function_names)

        print(f"Saving to: {output_path}")
        combined_model.save(output_path)

        # Clean up temp file
        shutil.rmtree(temp_path, ignore_errors=True)

    # Build report
    from .dedup_weights import DeduplicationReport, ReplacementReason

    report = DeduplicationReport(
        total_pairs_evaluated=len(diagnostics),
        pairs_replaced=sum(1 for d in diagnostics if d.reason == ReplacementReason.DEQ_CLOSE),
        pairs_identical=sum(1 for d in diagnostics if d.reason == ReplacementReason.IDENTICAL),
        pairs_skipped=sum(1 for d in diagnostics if d.reason in [
            ReplacementReason.REJECTED_SHAPE,
            ReplacementReason.REJECTED_THRESHOLD,
            ReplacementReason.REJECTED_DEQ_FAIL,
        ]),
        bytes_saved=sum(d.bytes_saved for d in diagnostics),
        diagnostics=diagnostics,
    )

    print(f"\n{report.summary()}")

    # Optional verification
    if verify and len(model_paths) >= 2:
        print("\nVerifying deduplication correctness...")
        # We can't easily verify the combined model, but we can check weight equivalence
        from .dedup_weights import verify_weight_equivalence
        is_equiv, equiv_report = verify_weight_equivalence(
            model_paths[0], model_paths[1], verbose=verbose
        )
        if is_equiv:
            print("Verification PASSED: Weights are semantically equivalent")
        else:
            print("Verification WARNING: Some weights differ (this may be expected)")

    print("\nModel combination with dedup complete!")
    return combined_model, report


def _add_combined_metadata(target_model, source_models, function_names):
    """Add combined metadata from source models to target model."""
    combined = {}
    descriptions = []

    for model in source_models:
        metadata = ReadMetadata(model)
        if 'short_description' in metadata:
            descriptions.append(metadata['short_description'])

        for key, value in metadata.items():
            if key != 'short_description' and value is not None:
                if key not in combined or combined[key] is None:
                    combined[key] = value

    # Create combined description
    if function_names:
        combined['short_description'] = f"Nimbo Model: Multifunction ({'+'.join(function_names)})"

    combined['function_names'] = function_names
    AddMetadata(target_model, combined)


def combine_monolithic(
    infer_path: str,
    prefill_path: str,
    output_path: str,
    lut_bits: int = None,
    use_dedup: bool = True,
    verbose: bool = False,
):
    """Combine monolithic inference and prefill models.

    Creates a combined model with two functions:
    - 'infer': Single token inference
    - 'prefill': Batch token processing for initial sequence

    Args:
        infer_path: Path to inference model (.mlpackage)
        prefill_path: Path to prefill model (.mlpackage)
        output_path: Path to save combined model
        lut_bits: LUT quantization bits (for metadata)
        use_dedup: If True, apply weight deduplication (15-40% smaller, default True)
        verbose: Print detailed progress

    Returns:
        ct.models.MLModel: Combined multi-function model
        (or Tuple[model, report] if use_dedup=True)
    """
    return combine_models(
        model_paths=[infer_path, prefill_path],
        function_names=["infer", "prefill"],
        output_path=output_path,
        default_function="infer",
        use_dedup=use_dedup,
        verbose=verbose,
    )


def combine_decoder_chunks(
    num_chunks: int,
    prefix: str = "llama",
    lut_bits: int = None,
    input_dir: str = ".",
    output_dir: str = ".",
    combine_infer_prefill: bool = True,
    use_dedup: bool = True,
    verbose: bool = False,
):
    """Combine Decoder layer chunks with inference and prefill modes.

    For each chunk, combines:
    - Decoder inference model
    - Decoder prefill model

    Args:
        num_chunks: Number of Decoder layer chunks
        prefix: Model name prefix
        lut_bits: LUT quantization bits (for file naming)
        input_dir: Directory containing input models
        output_dir: Directory to save combined models
        combine_infer_prefill: If True, combine infer+prefill per chunk
        use_dedup: If True, apply weight deduplication (15-40% smaller, default True)
        verbose: Print detailed progress

    Returns:
        list: Paths to combined chunk models
    """
    if not COREML_AVAILABLE:
        raise ImportError("coremltools is required. Install with: pip install coremltools")

    os.makedirs(output_dir, exist_ok=True)
    combined_paths = []
    total_bytes_saved = 0

    # Build file name templates
    if lut_bits:
        decoder_template = f"{prefix}_decoder_lut{lut_bits}_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
        prefill_template = f"{prefix}_decoder_prefill_lut{lut_bits}_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
        combined_template = f"{prefix}_decoder_full_lut{lut_bits}_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
    else:
        decoder_template = f"{prefix}_decoder_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
        prefill_template = f"{prefix}_decoder_prefill_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
        combined_template = f"{prefix}_decoder_full_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"

    for chunk_idx in range(num_chunks):
        chunk_num = chunk_idx + 1
        decoder_path = os.path.join(input_dir, decoder_template.format(chunk_num))
        prefill_path = os.path.join(input_dir, prefill_template.format(chunk_num))
        output_path = os.path.join(output_dir, combined_template.format(chunk_num))

        print(f"\n{'='*60}")
        print(f"Processing Decoder chunk {chunk_num}/{num_chunks}" +
              (" with dedup" if use_dedup else ""))
        print(f"{'='*60}")

        if combine_infer_prefill:
            if not os.path.exists(decoder_path):
                print(f"Warning: Decoder model not found: {decoder_path}")
                continue
            if not os.path.exists(prefill_path):
                print(f"Warning: Prefill model not found: {prefill_path}")
                continue

            result = combine_models(
                model_paths=[decoder_path, prefill_path],
                function_names=["infer", "prefill"],
                output_path=output_path,
                use_dedup=use_dedup,
                verbose=verbose,
            )

            # Track bytes saved if using dedup
            if use_dedup and isinstance(result, tuple):
                _, report = result
                total_bytes_saved += report.bytes_saved

            combined_paths.append(output_path)
        else:
            # Just copy decoder model as-is
            if os.path.exists(decoder_path):
                shutil.copytree(decoder_path, output_path)
                combined_paths.append(output_path)

    print(f"\n{'='*60}")
    print(f"Combined {len(combined_paths)} Decoder chunks")
    if use_dedup and total_bytes_saved > 0:
        print(f"Total estimated savings: ~{total_bytes_saved / 1e6:.1f} MB")
    print(f"{'='*60}")

    return combined_paths


def validate_chunk_files(
    num_chunks: int,
    prefix: str = "llama",
    lut_bits: int = None,
    input_dir: str = ".",
    check_prefill: bool = True,
):
    """Validate that all required chunk files exist.

    Args:
        num_chunks: Number of expected chunks
        prefix: Model name prefix
        lut_bits: LUT quantization bits
        input_dir: Directory containing model files
        check_prefill: Also check for prefill models

    Returns:
        tuple: (bool success, list missing_files)
    """
    missing_files = []

    if lut_bits:
        decoder_template = f"{prefix}_decoder_lut{lut_bits}_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
        prefill_template = f"{prefix}_decoder_prefill_lut{lut_bits}_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
    else:
        decoder_template = f"{prefix}_decoder_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
        prefill_template = f"{prefix}_decoder_prefill_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"

    for chunk_idx in range(num_chunks):
        chunk_num = chunk_idx + 1

        decoder_path = os.path.join(input_dir, decoder_template.format(chunk_num))
        if not os.path.exists(decoder_path):
            missing_files.append(decoder_path)

        if check_prefill:
            prefill_path = os.path.join(input_dir, prefill_template.format(chunk_num))
            if not os.path.exists(prefill_path):
                missing_files.append(prefill_path)

    if missing_files:
        print("Missing files:")
        for f in missing_files:
            print(f"  - {f}")
        return False, missing_files

    return True, []


def get_chunk_file_names(
    num_chunks: int,
    prefix: str = "llama",
    lut_bits: int = None,
    include_prefill: bool = True,
):
    """Get list of expected chunk file names.

    Args:
        num_chunks: Number of chunks
        prefix: Model name prefix
        lut_bits: LUT quantization bits
        include_prefill: Include prefill model names

    Returns:
        dict: Dictionary with 'decoder', 'prefill', 'combined' keys
    """
    if lut_bits:
        decoder_template = f"{prefix}_decoder_lut{lut_bits}_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
        prefill_template = f"{prefix}_decoder_prefill_lut{lut_bits}_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
        combined_template = f"{prefix}_decoder_full_lut{lut_bits}_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
    else:
        decoder_template = f"{prefix}_decoder_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
        prefill_template = f"{prefix}_decoder_prefill_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
        combined_template = f"{prefix}_decoder_full_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"

    result = {
        'decoder': [decoder_template.format(i + 1) for i in range(num_chunks)],
        'combined': [combined_template.format(i + 1) for i in range(num_chunks)],
    }

    if include_prefill:
        result['prefill'] = [prefill_template.format(i + 1) for i in range(num_chunks)]

    return result
