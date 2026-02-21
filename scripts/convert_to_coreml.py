#!/usr/bin/env python3
# Copyright (c) 2025, Nimbo
# Licensed under MIT License

"""
HuggingFace to CoreML Conversion CLI

Convert HuggingFace LLaMA models to CoreML format for Apple Neural Engine.

Usage:
    # Simple conversion
    python scripts/convert_to_coreml.py --model meta-llama/Llama-3.2-1B --output ./output

    # With options
    python scripts/convert_to_coreml.py \\
        --model meta-llama/Llama-3.2-3B \\
        --output ./output \\
        --lut-bits 4 \\
        --context-length 512 \\
        --split-chunks 4

    # From local model
    python scripts/convert_to_coreml.py --model ./my_model --output ./output
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace LLaMA models to CoreML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert LLaMA 3.2 1B with 4-bit quantization
  python scripts/convert_to_coreml.py \\
      --model meta-llama/Llama-3.2-1B \\
      --output ./coreml_models \\
      --lut-bits 4

  # Convert with model splitting for large models
  python scripts/convert_to_coreml.py \\
      --model meta-llama/Llama-3.2-3B \\
      --output ./coreml_models \\
      --lut-bits 4 \\
      --split-chunks 4

  # Convert local fine-tuned model
  python scripts/convert_to_coreml.py \\
      --model ./my_finetuned_model \\
      --output ./coreml_models \\
      --context-length 1024

Supported Models:
  - LLaMA 3.2 (1B, 3B)
  - LLaMA 3 (8B, 70B)
  - LLaMA 2 (7B, 13B, 70B)
  - Any LLaMA-architecture model

Output:
  The converter will create .mlpackage files optimized for
  Apple Neural Engine (ANE) execution on iOS and macOS devices.
        """
    )

    # Required arguments
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., meta-llama/Llama-3.2-1B) or local path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for CoreML models"
    )

    # Quantization
    parser.add_argument(
        "--lut-bits",
        type=int,
        default=4,
        choices=[4, 6, 8],
        help="LUT quantization bits (default: 4)"
    )

    # Context settings
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Maximum context length (default: 512)"
    )
    parser.add_argument(
        "--state-length",
        type=int,
        default=None,
        help="KV cache state length (default: same as context-length)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for prefill mode (default: 64)"
    )

    # Model splitting
    parser.add_argument(
        "--split-chunks",
        type=int,
        default=1,
        help="Number of chunks to split decoder layers (default: 1 = no split)"
    )

    # Output options
    parser.add_argument(
        "--no-prefill",
        action="store_true",
        help="Skip prefill mode conversion"
    )
    parser.add_argument(
        "--no-combine",
        action="store_true",
        help="Don't combine infer+prefill into multi-function model"
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable weight deduplication (larger file size)"
    )

    # Authentication
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token for gated models"
    )

    # Misc
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Check for coremltools
    try:
        import coremltools
    except ImportError:
        print("ERROR: coremltools is required but not installed.")
        print("This package requires macOS. Install with: pip install coremltools")
        return 1

    # Import converter
    try:
        from nimbo.export.coreml.hf_converter import convert_hf_to_coreml, ConversionConfig
    except ImportError as e:
        print(f"ERROR: Failed to import Nimbo converter: {e}")
        return 1

    # Set state_length
    state_length = args.state_length or args.context_length

    # Create config
    config = ConversionConfig(
        context_length=args.context_length,
        state_length=state_length,
        batch_size=args.batch_size,
        lut_bits=args.lut_bits,
        split_model=args.split_chunks > 1,
        num_chunks=args.split_chunks,
        include_prefill=not args.no_prefill,
        combine_models=not args.no_combine,
        use_dedup=not args.no_dedup,
    )

    print("=" * 60)
    print("Nimbo HuggingFace to CoreML Converter")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"LUT bits: {args.lut_bits}")
    print(f"Context length: {args.context_length}")
    print(f"Chunks: {args.split_chunks}")
    print()

    # Run conversion
    result = convert_hf_to_coreml(
        model_id_or_path=args.model,
        output_dir=args.output,
        token=args.token,
        config=config,
    )

    # Print result
    if result.success:
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print("\nOutput files:")
        for path in result.output_paths:
            print(f"  - {path}")
        return 0
    else:
        print("\n" + "=" * 60)
        print("FAILED!")
        print("=" * 60)
        print("\nErrors:")
        for error in result.errors:
            print(f"  {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
