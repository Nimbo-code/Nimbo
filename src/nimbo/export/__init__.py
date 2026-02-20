# Copyright (c) 2025, Nimbo
# Licensed under MIT License

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

from .coreml import (
    LlamaConverter,
    LlamaConfig,
    LlamaForCausalLM,
    BaseConverter,
)

__all__ = [
    "LlamaConverter",
    "LlamaConfig",
    "LlamaForCausalLM",
    "BaseConverter",
]
