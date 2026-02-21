# Copyright (c) 2025, Nimbo
# Licensed under the Apache License, Version 2.0
# Based on Anemll (https://github.com/Anemll/Anemll) - MIT License

"""
CoreML model metadata utilities for Nimbo.

This module provides functions to add, read, and combine metadata
for CoreML models converted by Nimbo.
"""

from enum import Enum
import os

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError


class ModelPart(Enum):
    """Enumeration of model parts for split conversion."""
    EMBEDDINGS = "1"
    FFN = "2"
    PREFILL = "2_prefill"
    LM_HEAD = "3"
    FULL = "123"
    MONOLITHIC = "monolithic"
    MONOLITHIC_PREFILL = "monolithic_prefill"


def get_nimbo_version():
    """Get Nimbo version from package metadata.

    Returns:
        str: The version string of the installed Nimbo package
    """
    try:
        package_version = version('nimbo')
        if package_version:
            return package_version
    except PackageNotFoundError:
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            pkg_info_paths = [
                os.path.join(current_dir, '..', '..', '..', '..', 'PKG-INFO'),
                os.path.join(current_dir, '..', '..', '..', '..', 'nimbo.egg-info', 'PKG-INFO'),
                os.path.join(current_dir, '..', '..', '..', 'PKG-INFO'),
            ]

            for pkg_info_path in pkg_info_paths:
                if os.path.exists(pkg_info_path):
                    with open(pkg_info_path, 'r') as f:
                        for line in f:
                            if line.startswith('Version:'):
                                return line.split(':')[1].strip()
        except Exception:
            pass

    return "0.0.4"  # Default version


def AddMetadata(model, params=None):
    """Add metadata to a CoreML model.

    Args:
        model: CoreML model to add metadata to
        params: Dictionary containing metadata parameters:
            - context_length: Context length used
            - num_chunks: Total number of chunks
            - chunk_no: Current chunk number
            - batch_size: Batch size (for prefill)
            - lut_bits: LUT quantization bits
            - split_part: Model part identifier
            - argmax_in_model: Whether argmax is computed in model
            - vocab_size: Vocabulary size
            - lm_head_chunk_sizes: Sizes of LM head chunks
    """
    nimbo_version = get_nimbo_version()

    if not hasattr(model, 'user_defined_metadata'):
        model.user_defined_metadata = {}

    if not hasattr(model, 'author'):
        model.author = ""
    if not hasattr(model, 'version'):
        model.version = ""
    if not hasattr(model, 'short_description'):
        model.short_description = ""

    model.author = f"Converted with Nimbo v{nimbo_version}"
    model.version = nimbo_version
    model.user_defined_metadata["com.nimbo.info"] = f"Converted with Nimbo v{nimbo_version}"

    if params:
        if 'short_description' in params:
            model.short_description = params['short_description']

        if 'context_length' in params and params['context_length'] is not None:
            model.user_defined_metadata["com.nimbo.context_length"] = str(params['context_length'])

        if 'num_chunks' in params and params['num_chunks'] is not None:
            model.user_defined_metadata["com.nimbo.num_chunks"] = str(params['num_chunks'])

        if 'chunk_no' in params and params['chunk_no'] is not None:
            model.user_defined_metadata["com.nimbo.chunk_no"] = str(params['chunk_no'])

        if 'batch_size' in params and params['batch_size'] is not None:
            model.user_defined_metadata["com.nimbo.batch_size"] = str(params['batch_size'])

        if 'lut_bits' in params and params['lut_bits'] is not None:
            model.user_defined_metadata["com.nimbo.lut_bits"] = str(params['lut_bits'])

        if 'argmax_in_model' in params and params['argmax_in_model']:
            model.user_defined_metadata["com.nimbo.argmax_in_model"] = "true"

        if 'vocab_size' in params and params['vocab_size'] is not None:
            model.user_defined_metadata["com.nimbo.vocab_size"] = str(params['vocab_size'])

        if 'lm_head_chunk_sizes' in params and params['lm_head_chunk_sizes'] is not None:
            chunk_sizes = params['lm_head_chunk_sizes']
            if isinstance(chunk_sizes, (list, tuple)):
                chunk_sizes = ",".join(str(int(x)) for x in chunk_sizes)
            model.user_defined_metadata["com.nimbo.lm_head_chunk_sizes"] = str(chunk_sizes)

        if 'function_names' in params and params['function_names'] is not None:
            model.short_description = f"Combined model with functions: {', '.join(params['function_names'])}"
            model.user_defined_metadata["com.nimbo.functions"] = ",".join(params['function_names'])

        if 'split_part' in params and params['split_part'] is not None:
            split_part = params['split_part']
            descriptions = {
                ModelPart.EMBEDDINGS.value: "Nimbo Model (Embeddings) converted to CoreML",
                ModelPart.PREFILL.value: "Nimbo Model (Prefill) converted to CoreML",
                ModelPart.LM_HEAD.value: "Nimbo Model (LM Head) converted to CoreML",
                ModelPart.FFN.value: "Nimbo Model (FFN) converted to CoreML",
                ModelPart.FULL.value: "Nimbo Model (Full) converted to CoreML",
                ModelPart.MONOLITHIC.value: "Nimbo Model (Monolithic) converted to CoreML",
                ModelPart.MONOLITHIC_PREFILL.value: "Nimbo Model (Monolithic Prefill) converted to CoreML",
            }
            model.short_description = descriptions.get(
                split_part, f"Nimbo Model Part {split_part} converted to CoreML"
            )


def ReadMetadata(model):
    """Read metadata from a CoreML model.

    Args:
        model: CoreML model to read metadata from

    Returns:
        dict: Dictionary containing metadata parameters
    """
    metadata = {}

    if not hasattr(model, 'user_defined_metadata'):
        model.user_defined_metadata = {}
    if not hasattr(model, 'author'):
        model.author = ""
    if not hasattr(model, 'version'):
        model.version = ""
    if not hasattr(model, 'short_description'):
        model.short_description = ""

    metadata['author'] = model.author
    metadata['version'] = model.version
    metadata['short_description'] = model.short_description

    for key, value in model.user_defined_metadata.items():
        if key.startswith('com.nimbo.'):
            clean_key = key.replace('com.nimbo.', '')
            metadata[clean_key] = value

    return metadata


def CombineMetadata(models):
    """Combine metadata from multiple models.

    Args:
        models: List of CoreML models

    Returns:
        dict: Combined metadata parameters
    """
    combined = {}

    for model in models:
        metadata = ReadMetadata(model)

        for key, value in metadata.items():
            if value is not None and (key not in combined or combined[key] is None):
                combined[key] = value

        if 'version' in metadata and metadata['version'] is not None:
            if 'version' not in combined or metadata['version'] > combined['version']:
                combined['version'] = metadata['version']

        if 'functions' in metadata and metadata['functions'] is not None:
            if 'functions' not in combined:
                combined['functions'] = []
            combined['functions'].extend(metadata['functions'].split(','))
            combined['functions'] = list(set(combined['functions']))

    return combined
