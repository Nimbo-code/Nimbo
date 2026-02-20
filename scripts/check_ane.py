#!/usr/bin/env python3
# Copyright (c) 2025, Nimbo
# Licensed under MIT License

"""
ANE Compatibility Checker CLI

Check if a model is compatible with Apple Neural Engine (ANE).

Usage:
    # Check PyTorch model from HuggingFace
    python scripts/check_ane.py --hf meta-llama/Llama-3.2-1B

    # Check PyTorch model from local path
    python scripts/check_ane.py --model ./model --output ane_report.txt

    # Check CoreML model
    python scripts/check_ane.py --coreml model.mlpackage --output ane_report.txt

    # Check our custom LlamaForCausalLM
    python scripts/check_ane.py --nimbo-config ./model/config.json
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def load_hf_model(model_id: str):
    """Load model from HuggingFace."""
    try:
        from transformers import AutoModel, AutoConfig
    except ImportError:
        print("ERROR: transformers library required")
        print("Install with: pip install transformers")
        sys.exit(1)

    print(f"Loading model from HuggingFace: {model_id}")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    return model


def load_local_model(model_path: str):
    """Load PyTorch model from local path."""
    try:
        from transformers import AutoModel, AutoConfig
    except ImportError:
        print("ERROR: transformers library required")
        sys.exit(1)

    print(f"Loading model from: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    return model


def load_nimbo_model(config_path: str):
    """Load Nimbo's LlamaForCausalLM from config."""
    try:
        from nimbo.export.coreml import LlamaConfig, LlamaForCausalLM
    except ImportError:
        print("ERROR: Could not import Nimbo export module")
        sys.exit(1)

    print(f"Loading Nimbo model from config: {config_path}")
    config = LlamaConfig.from_json(config_path)
    model = LlamaForCausalLM(config)

    # Try to load weights if available
    model_dir = os.path.dirname(config_path)
    try:
        model.load_pretrained_weights(model_dir)
        print("  Weights loaded successfully")
    except Exception as e:
        print(f"  Note: Could not load weights ({e})")
        print("  Continuing with architecture-only analysis")

    return model


def check_pytorch_model(model, output_path: str, verbose: bool):
    """Check PyTorch model for ANE compatibility."""
    from nimbo.export.coreml.ane_checker import ANEChecker

    checker = ANEChecker(verbose=verbose)
    report = checker.check_pytorch_model(model)

    # Print summary to console
    print("\n" + "=" * 60)
    print("ANE COMPATIBILITY REPORT")
    print("=" * 60)
    print(f"\nModel: {report.model_name}")
    print(f"ANE Compatibility Score: {report.get_compatibility_score():.1f}/100")
    print(f"\nLayer Statistics:")
    print(f"  Total:            {report.total_layers}")
    print(f"  Optimized (ANE):  {report.optimized_layers}")
    print(f"  Compatible:       {report.compatible_layers}")
    print(f"  Needs conversion: {report.needs_conversion_layers}")
    print(f"  Partial:          {report.partial_layers}")
    print(f"  Incompatible:     {report.incompatible_layers}")

    if report.summary.get("recommendation"):
        print(f"\nRecommendation: {report.summary['recommendation']}")

    # Show critical issues
    errors = [i for i in report.issues if i.level.value == "error"]
    warnings = [i for i in report.issues if i.level.value == "warning"]

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for issue in errors[:5]:
            print(f"  - [{issue.layer_name}] {issue.message}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for issue in warnings[:5]:
            print(f"  - [{issue.layer_name}] {issue.message}")
        if len(warnings) > 5:
            print(f"  ... and {len(warnings) - 5} more warnings")

    # Save report
    if output_path:
        checker.save_report(report, output_path)
        print(f"\nFull report saved to: {output_path}")

    return report


def check_coreml_model(model_path: str, output_path: str, verbose: bool):
    """Check CoreML model for ANE compatibility."""
    try:
        from nimbo.export.coreml.ane_checker import ANEChecker
    except ImportError:
        # Direct import fallback
        sys.path.insert(0, str(project_root / "src" / "nimbo" / "export" / "coreml"))
        from ane_checker import ANEChecker

    try:
        import coremltools
    except ImportError:
        print("ERROR: coremltools required for CoreML model checking")
        print("Install with: pip install coremltools")
        sys.exit(1)

    checker = ANEChecker(verbose=verbose)
    report = checker.check_coreml_model(model_path)

    # Print summary
    print("\n" + "=" * 60)
    print("ANE COMPATIBILITY REPORT")
    print("=" * 60)
    print(f"\nModel: {report.model_name}")
    print(f"ANE Compatibility Score: {report.get_compatibility_score():.1f}/100")

    # Save report
    if output_path:
        checker.save_report(report, output_path)
        print(f"\nFull report saved to: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Check model compatibility with Apple Neural Engine (ANE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check HuggingFace model
  python scripts/check_ane.py --hf meta-llama/Llama-3.2-1B --output report.txt

  # Check local PyTorch model
  python scripts/check_ane.py --model ./my_model --output report.txt

  # Check CoreML model
  python scripts/check_ane.py --coreml model.mlpackage --output report.txt

  # Check Nimbo model architecture
  python scripts/check_ane.py --nimbo-config ./model/config.json --output report.txt

Output:
  The report includes:
  - ANE compatibility score (0-100)
  - Layer-by-layer analysis
  - Issues with severity levels (error/warning/info)
  - Suggestions for improving ANE compatibility
        """
    )

    # Model input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--hf", type=str, metavar="MODEL_ID",
                             help="HuggingFace model ID (e.g., meta-llama/Llama-3.2-1B)")
    input_group.add_argument("--model", type=str, metavar="PATH",
                             help="Path to local PyTorch model directory")
    input_group.add_argument("--coreml", type=str, metavar="PATH",
                             help="Path to CoreML model (.mlpackage or .mlmodel)")
    input_group.add_argument("--nimbo-config", type=str, metavar="PATH",
                             help="Path to Nimbo config.json")

    # Output options
    parser.add_argument("--output", "-o", type=str,
                        help="Output report path (.txt or .json)")
    parser.add_argument("--format", choices=["txt", "json"], default="txt",
                        help="Output format (default: txt)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        if args.hf:
            name = args.hf.replace("/", "_")
        elif args.model:
            name = os.path.basename(args.model)
        elif args.coreml:
            name = os.path.basename(args.coreml).replace(".mlpackage", "").replace(".mlmodel", "")
        else:
            name = "nimbo_model"

        output_path = f"ane_report_{name}.{args.format}"

    print("=" * 60)
    print("Nimbo ANE Compatibility Checker")
    print("=" * 60)

    # Load and check model
    if args.hf:
        model = load_hf_model(args.hf)
        report = check_pytorch_model(model, output_path, args.verbose)

    elif args.model:
        model = load_local_model(args.model)
        report = check_pytorch_model(model, output_path, args.verbose)

    elif args.coreml:
        if not os.path.exists(args.coreml):
            print(f"ERROR: CoreML model not found: {args.coreml}")
            sys.exit(1)
        report = check_coreml_model(args.coreml, output_path, args.verbose)

    elif args.nimbo_config:
        if not os.path.exists(args.nimbo_config):
            print(f"ERROR: Config file not found: {args.nimbo_config}")
            sys.exit(1)
        model = load_nimbo_model(args.nimbo_config)
        report = check_pytorch_model(model, output_path, args.verbose)

    # Return code based on compatibility
    score = report.get_compatibility_score()
    if score >= 70:
        print("\nResult: Model is ANE compatible")
        return 0
    elif score >= 50:
        print("\nResult: Model has moderate ANE compatibility")
        return 0
    else:
        print("\nResult: Model has poor ANE compatibility")
        return 1


if __name__ == "__main__":
    sys.exit(main())
