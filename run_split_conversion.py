#!/usr/bin/env python3
"""Run split CoreML conversion with per-component quantization."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from nimbo.export.coreml.hf_converter import convert_hf_to_coreml

result = convert_hf_to_coreml(
    'final_merged',
    'coreml_models/split_lut6',
    lut_bits=6,
    lut_embeddings_bits=-1,   # float16 (no quantization)
    lut_lmhead_bits=6,
    split_model=True,
    num_chunks=1,
)

print(f"\nSuccess: {result.success}")
if result.errors:
    print(f"Errors: {result.errors}")
if result.output_paths:
    print(f"Output files: {result.output_paths}")
