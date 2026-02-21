# Copyright (c) 2025, Nimbo
# Licensed under the Apache License, Version 2.0

"""
Nimbo Export Module - Convert HuggingFace models to on-device formats.

Supported formats:
- CoreML: For Apple Neural Engine (ANE) deployment on iOS/macOS
- ONNX: For cross-platform deployment (coming soon)

Supported models:
- LLaMA 3.2 (1B, 3B)
- LLaMA 3 (8B, 70B)
- LLaMA 2 (7B, 13B, 70B)

Example usage:
    from nimbo.export import CoreMLConverter

    converter = CoreMLConverter(
        model_path="meta-llama/Llama-3.2-1B",
        context_length=512,
        lut_bits=4,  # 4-bit quantization
    )
    converter.convert(output_dir="./output")
"""

# CoreML is only available on macOS
# Make it optional for Linux/Windows users
try:
    from .coreml import (
        LlamaConverter,
        LlamaConfig,
        LlamaForCausalLM,
        BaseConverter,
    )
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    LlamaConverter = None
    LlamaConfig = None
    LlamaForCausalLM = None
    BaseConverter = None

__all__ = [
    "LlamaConverter",
    "LlamaConfig",
    "LlamaForCausalLM",
    "BaseConverter",
    "COREML_AVAILABLE",
]
