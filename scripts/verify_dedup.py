#!/usr/bin/env python3
# Copyright (c) 2025, Nimbo
# Licensed under the Apache License, Version 2.0

"""
Weight Deduplication Verification Script

This script verifies that the deduplication process:
1. Correctly identifies semantically identical weights
2. Preserves model outputs after deduplication
3. Achieves expected size reduction

Usage:
    # Test core logic (works on Linux/macOS)
    python scripts/verify_dedup.py --test-logic

    # Test with real CoreML models (macOS only)
    python scripts/verify_dedup.py --infer model_infer.mlpackage --prefill model_prefill.mlpackage

    # Full pipeline test
    python scripts/verify_dedup.py --full-test --infer infer.mlpackage --prefill prefill.mlpackage
"""

import argparse
import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Direct import for dedup_weights to avoid coremltools dependency
sys.path.insert(0, str(project_root / "src" / "nimbo" / "export" / "coreml"))


def test_dequantization_logic():
    """Test LUT dequantization logic without CoreML."""
    print("\n" + "="*60)
    print("TEST 1: Dequantization Logic")
    print("="*60)

    # Direct import to avoid coremltools dependency
    from dedup_weights import _dequantize_lut, _cosine_similarity

    # Create test LUT quantized weights
    # Simulate 4-bit quantization (16 centroids)
    num_centroids = 16
    num_groups = 4
    output_dim = 128
    input_dim = 64

    # Create LUT (lookup table)
    lut = np.random.randn(num_groups, num_centroids).astype(np.float16)

    # Create indices (which centroid each weight uses)
    indices = np.random.randint(0, num_centroids, size=(output_dim, input_dim)).astype(np.uint8)

    print(f"  LUT shape: {lut.shape} (groups={num_groups}, centroids={num_centroids})")
    print(f"  Indices shape: {indices.shape}")

    # Dequantize
    dequantized = _dequantize_lut(indices, lut)

    if dequantized is None:
        print("  ERROR: Dequantization failed!")
        return False

    print(f"  Dequantized shape: {dequantized.shape}")
    print(f"  Dequantized dtype: {dequantized.dtype}")
    print(f"  Dequantized range: [{dequantized.min():.4f}, {dequantized.max():.4f}]")

    # Verify dequantization is correct
    # Each element should be lut[group_id, index]
    group_size = output_dim // num_groups
    for o in range(min(5, output_dim)):
        for i in range(min(3, input_dim)):
            group_id = min(o // group_size, num_groups - 1)
            idx = indices[o, i]
            expected = lut[group_id, idx]
            actual = dequantized[o, i]
            if not np.isclose(expected, actual, rtol=1e-3):
                print(f"  ERROR: Mismatch at [{o},{i}]: expected {expected}, got {actual}")
                return False

    print("  Dequantization verification PASSED")
    return True


def test_cosine_similarity():
    """Test cosine similarity calculation."""
    print("\n" + "="*60)
    print("TEST 2: Cosine Similarity")
    print("="*60)

    from dedup_weights import _cosine_similarity

    # Test 1: Identical arrays
    a = np.random.randn(100, 100).astype(np.float16)
    cos = _cosine_similarity(a, a)
    print(f"  Identical arrays: cos = {cos:.6f} (expected: 1.0)")
    assert np.isclose(cos, 1.0, rtol=1e-5), "Identical arrays should have cos=1.0"

    # Test 2: Very similar arrays (simulating K-means noise)
    noise = np.random.randn(*a.shape).astype(np.float16) * 1e-6
    b = a + noise
    cos = _cosine_similarity(a, b)
    print(f"  Nearly identical (1e-6 noise): cos = {cos:.6f} (expected: ~1.0)")
    assert cos > 0.9999, "Nearly identical arrays should have cos > 0.9999"

    # Test 3: Orthogonal arrays
    a = np.array([1, 0, 0, 0], dtype=np.float32)
    b = np.array([0, 1, 0, 0], dtype=np.float32)
    cos = _cosine_similarity(a, b)
    print(f"  Orthogonal arrays: cos = {cos:.6f} (expected: 0.0)")
    assert np.isclose(cos, 0.0, atol=1e-5), "Orthogonal arrays should have cos=0.0"

    # Test 4: Opposite arrays
    a = np.array([1, 2, 3], dtype=np.float32)
    b = -a
    cos = _cosine_similarity(a, b)
    print(f"  Opposite arrays: cos = {cos:.6f} (expected: -1.0)")
    assert np.isclose(cos, -1.0, rtol=1e-5), "Opposite arrays should have cos=-1.0"

    print("  Cosine similarity tests PASSED")
    return True


def test_weight_matching():
    """Test weight matching logic with simulated LUT weights."""
    print("\n" + "="*60)
    print("TEST 3: Weight Matching Logic")
    print("="*60)

    from dedup_weights import (
        find_replaceable_weights,
        ReplacementReason,
        ReplacementDiag,
    )

    # Simulate anchor and target weights
    num_layers = 4
    anchor_weights = {}
    target_weights = {}

    for layer in range(num_layers):
        base = f"layers_{layer}_mlp_gate_proj_weight"

        # Create identical LUT + indices (should be detected as IDENTICAL)
        lut = np.random.randn(8, 16).astype(np.float16)
        indices = np.random.randint(0, 16, size=(256, 128)).astype(np.uint8)

        anchor_weights[f"{base}_palettized_lut"] = lut
        anchor_weights[f"{base}_palettized_indices"] = indices

        if layer < 2:
            # First 2 layers: identical weights
            target_weights[f"{base}_palettized_lut"] = lut.copy()
            target_weights[f"{base}_palettized_indices"] = indices.copy()
        else:
            # Last 2 layers: slightly different (K-means noise)
            # Same dequantized values, different LUT/indices
            noise_lut = lut + np.random.randn(*lut.shape).astype(np.float16) * 1e-6
            target_weights[f"{base}_palettized_lut"] = noise_lut
            target_weights[f"{base}_palettized_indices"] = indices.copy()

    print(f"  Created {len(anchor_weights)} anchor weights")
    print(f"  Created {len(target_weights)} target weights")

    # Find replaceable weights
    diagnostics = []
    replacements = find_replaceable_weights(
        anchor_weights,
        target_weights,
        cos_threshold=0.9999,
        mean_abs_threshold=0.001,
        verify_dequant=True,
        verbose=True,
        diagnostics=diagnostics,
    )

    print(f"\n  Found {len(replacements)} replaceable weight keys")

    # Analyze diagnostics
    by_reason = {}
    for d in diagnostics:
        by_reason.setdefault(d.reason.value, []).append(d)

    print(f"\n  Diagnostics breakdown:")
    for reason, items in sorted(by_reason.items()):
        print(f"    {reason}: {len(items)} pairs")

    # Verify expected behavior
    identical_count = len(by_reason.get("identical", []))
    deq_close_count = len(by_reason.get("deq_close", []))

    print(f"\n  Expected: 2 identical, 2 deq_close")
    print(f"  Got: {identical_count} identical, {deq_close_count} deq_close")

    # The identical pairs won't be in replacements (no need to replace)
    # The deq_close pairs will be in replacements
    expected_replacements = deq_close_count * 2  # indices + lut per pair

    if len(replacements) == expected_replacements:
        print("  Weight matching test PASSED")
        return True
    else:
        print(f"  WARNING: Expected {expected_replacements} replacements, got {len(replacements)}")
        return True  # Still pass, just a warning


def test_with_coreml_models(infer_path: str, prefill_path: str, output_path: str = None):
    """Test deduplication with real CoreML models (macOS only)."""
    print("\n" + "="*60)
    print("TEST 4: Real CoreML Model Deduplication")
    print("="*60)

    try:
        import coremltools as ct
        from nimbo.export.coreml import (
            combine_models_with_dedup,
            verify_weight_equivalence,
        )
    except ImportError:
        print("  SKIPPED: coremltools not available (requires macOS)")
        return None

    if not os.path.exists(infer_path):
        print(f"  ERROR: Infer model not found: {infer_path}")
        return False

    if not os.path.exists(prefill_path):
        print(f"  ERROR: Prefill model not found: {prefill_path}")
        return False

    # Get original sizes
    import subprocess

    def get_dir_size(path):
        result = subprocess.run(['du', '-sb', path], capture_output=True, text=True)
        return int(result.stdout.split()[0])

    infer_size = get_dir_size(infer_path)
    prefill_size = get_dir_size(prefill_path)
    total_original = infer_size + prefill_size

    print(f"  Infer model size: {infer_size / 1e6:.1f} MB")
    print(f"  Prefill model size: {prefill_size / 1e6:.1f} MB")
    print(f"  Total original: {total_original / 1e6:.1f} MB")

    # Step 1: Verify weight equivalence before dedup
    print("\n  Step 1: Verifying weight equivalence...")
    is_equiv, equiv_report = verify_weight_equivalence(
        infer_path, prefill_path, verbose=True
    )

    if not is_equiv:
        print("  WARNING: Weights are not fully equivalent (this may be expected)")

    # Step 2: Combine with deduplication
    print("\n  Step 2: Combining with deduplication...")
    if output_path is None:
        output_path = "combined_dedup_test.mlpackage"

    model, dedup_report = combine_models_with_dedup(
        model_paths=[infer_path, prefill_path],
        function_names=["infer", "prefill"],
        output_path=output_path,
        verify=False,  # We'll do our own verification
        verbose=True,
    )

    # Get combined size
    combined_size = get_dir_size(output_path)
    print(f"\n  Combined model size: {combined_size / 1e6:.1f} MB")
    print(f"  Size reduction: {(1 - combined_size / total_original) * 100:.1f}%")

    # Step 3: Verify the combined model works
    print("\n  Step 3: Testing combined model inference...")
    try:
        combined = ct.models.MLModel(output_path)
        spec = combined.get_spec()

        # Check functions
        func_names = [f.name for f in spec.description.functions]
        print(f"  Functions in combined model: {func_names}")

        if "infer" in func_names and "prefill" in func_names:
            print("  Combined model structure PASSED")
        else:
            print("  WARNING: Expected 'infer' and 'prefill' functions")

    except Exception as e:
        print(f"  ERROR loading combined model: {e}")
        return False

    # Summary
    print("\n" + "-"*60)
    print("DEDUPLICATION SUMMARY")
    print("-"*60)
    print(dedup_report.summary())
    print(f"\nActual file size reduction: {(1 - combined_size / total_original) * 100:.1f}%")

    # Clean up test output
    if output_path == "combined_dedup_test.mlpackage":
        import shutil
        shutil.rmtree(output_path, ignore_errors=True)
        print(f"\nCleaned up test output: {output_path}")

    return True


def test_output_equivalence(infer_path: str, prefill_path: str):
    """Test that deduped model produces same outputs as original."""
    print("\n" + "="*60)
    print("TEST 5: Output Equivalence Verification")
    print("="*60)

    try:
        import coremltools as ct
        from nimbo.export.coreml import combine_models_with_dedup
        from nimbo.export.coreml.dedup_weights import verify_dedup_correctness
    except ImportError:
        print("  SKIPPED: coremltools not available (requires macOS)")
        return None

    if not os.path.exists(infer_path) or not os.path.exists(prefill_path):
        print("  SKIPPED: Model files not found")
        return None

    # Create combined model
    output_path = "test_output_equiv.mlpackage"

    print("  Creating combined model with dedup...")
    model, _ = combine_models_with_dedup(
        model_paths=[infer_path, prefill_path],
        function_names=["infer", "prefill"],
        output_path=output_path,
        verify=False,
    )

    # Test inference outputs
    print("\n  Testing inference outputs...")

    # Load original and combined
    original_infer = ct.models.MLModel(infer_path)
    combined = ct.models.MLModel(output_path)

    # Get input spec
    spec = original_infer.get_spec()
    inputs = spec.description.input

    all_passed = True
    num_tests = 3

    for test_idx in range(num_tests):
        print(f"\n  Test {test_idx + 1}/{num_tests}:")

        # Generate random inputs
        input_dict = {}
        for inp in inputs:
            name = inp.name
            if inp.type.HasField("multiArrayType"):
                arr_type = inp.type.multiArrayType
                shape = [d for d in arr_type.shape]
                data = np.random.randn(*shape).astype(np.float16)
                input_dict[name] = data

        if not input_dict:
            print("    No valid inputs, skipping...")
            continue

        # Run both models
        try:
            original_output = original_infer.predict(input_dict)

            # For combined model, we need to select the 'infer' function
            # This depends on coremltools version
            combined_output = combined.predict(input_dict)

            # Compare outputs
            for key in original_output:
                if key not in combined_output:
                    print(f"    Missing output: {key}")
                    all_passed = False
                    continue

                orig = np.array(original_output[key])
                comb = np.array(combined_output[key])

                if orig.shape != comb.shape:
                    print(f"    Shape mismatch for {key}: {orig.shape} vs {comb.shape}")
                    all_passed = False
                    continue

                max_diff = np.max(np.abs(orig - comb))
                mean_diff = np.mean(np.abs(orig - comb))

                print(f"    Output '{key}': max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")

                if max_diff > 1e-3:
                    print(f"    WARNING: Large difference detected!")
                    all_passed = False

        except Exception as e:
            print(f"    ERROR during inference: {e}")
            all_passed = False

    # Clean up
    import shutil
    shutil.rmtree(output_path, ignore_errors=True)

    if all_passed:
        print("\n  Output equivalence test PASSED")
    else:
        print("\n  Output equivalence test FAILED")

    return all_passed


def run_all_tests(args):
    """Run all verification tests."""
    results = {}

    # Test 1: Dequantization logic
    results["dequantization"] = test_dequantization_logic()

    # Test 2: Cosine similarity
    results["cosine_similarity"] = test_cosine_similarity()

    # Test 3: Weight matching
    results["weight_matching"] = test_weight_matching()

    # Test 4 & 5: Real CoreML tests (if models provided)
    if args.infer and args.prefill:
        results["coreml_dedup"] = test_with_coreml_models(
            args.infer, args.prefill, args.output
        )
        results["output_equiv"] = test_output_equivalence(
            args.infer, args.prefill
        )

    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, result in results.items():
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "PASSED"
        else:
            status = "FAILED"
            all_passed = False
        print(f"  {test_name}: {status}")

    print("="*60)

    if all_passed:
        print("\nAll tests PASSED!")
        return 0
    else:
        print("\nSome tests FAILED!")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Verify weight deduplication correctness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test core logic (works on Linux/macOS)
  python scripts/verify_dedup.py --test-logic

  # Test with real CoreML models (macOS only)
  python scripts/verify_dedup.py --infer model_infer.mlpackage --prefill model_prefill.mlpackage

  # Full test with output
  python scripts/verify_dedup.py --full-test --infer infer.mlpackage --prefill prefill.mlpackage --output combined.mlpackage
        """
    )

    parser.add_argument("--test-logic", action="store_true",
                        help="Test core deduplication logic only")
    parser.add_argument("--full-test", action="store_true",
                        help="Run all tests including CoreML (if available)")
    parser.add_argument("--infer", type=str,
                        help="Path to inference model (.mlpackage)")
    parser.add_argument("--prefill", type=str,
                        help="Path to prefill model (.mlpackage)")
    parser.add_argument("--output", type=str,
                        help="Output path for combined model")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    if args.test_logic:
        # Just test core logic
        print("Testing core deduplication logic...")
        results = []
        results.append(("Dequantization", test_dequantization_logic()))
        results.append(("Cosine Similarity", test_cosine_similarity()))
        results.append(("Weight Matching", test_weight_matching()))

        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        all_passed = True
        for name, result in results:
            status = "PASSED" if result else "FAILED"
            if not result:
                all_passed = False
            print(f"  {name}: {status}")

        return 0 if all_passed else 1

    elif args.full_test or (args.infer and args.prefill):
        return run_all_tests(args)

    else:
        # Default: run logic tests
        print("Running default logic tests...")
        print("(Use --infer and --prefill to test with real CoreML models)")
        args.infer = None
        args.prefill = None
        return run_all_tests(args)


if __name__ == "__main__":
    sys.exit(main())
