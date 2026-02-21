# Copyright (c) 2025, Nimbo
# Licensed under the Apache License, Version 2.0
# Based on Anemll (https://github.com/Anemll/Anemll) - MIT License

"""Surgical weight deduplication for CoreML multifunction models.

Before calling ct.utils.save_multifunction(), this utility replaces palettized
weight blobs (LUT + indices) in non-anchor models with the anchor model's blobs
when the dequantized values are semantically identical. This forces byte-identical
const blobs so CoreMLTools' dedup pass can share them.

Background:
  MIL optimization passes (add_fp16_cast, constant folding) produce microscopically
  different fp16 representations depending on graph shape (context length, seq_len).
  K-means then converges to different LUT centroids + index assignments that encode
  the *same* dequantized values (verified: cosine similarity = 1.0, mean_diff ~3e-8).
  CoreMLTools dedup compares raw bytes, so these semantically-identical weights are
  not shared — causing ~15-40% size bloat depending on the function combination.

Usage:
    from nimbo.export.coreml.dedup_weights import prepare_dedup_sources

    sources = [(infer_path, "main", "infer"), (prefill_path, "main", "prefill")]
    with prepare_dedup_sources(sources) as deduped:
        desc = ct.utils.MultiFunctionDescriptor()
        for path, src_fn, tgt_fn in deduped:
            desc.add_function(path, src_fn, tgt_fn)
        desc.default_function_name = "infer"
        ct.utils.save_multifunction(desc, output_path)

Verification:
    from nimbo.export.coreml.dedup_weights import verify_dedup_correctness

    # Verify that deduplication preserves model outputs
    is_correct, report = verify_dedup_correctness(
        original_path="prefill.mlpackage",
        deduped_path="prefill_deduped.mlpackage",
    )
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Replacement reason codes
# ---------------------------------------------------------------------------

class ReplacementReason(Enum):
    """Reason code for each weight pair evaluation."""
    IDENTICAL = "identical"           # Already byte-identical, no action needed
    DEQ_CLOSE = "deq_close"           # Dequantized values match within thresholds
    REPLACED_NO_VERIFY = "replaced_no_verify"  # Replaced without dequant verification
    REJECTED_SHAPE = "rejected_shape"          # Shape mismatch
    REJECTED_THRESHOLD = "rejected_threshold"  # Failed acceptance thresholds
    REJECTED_DEQ_FAIL = "rejected_deq_fail"    # Dequantization failed
    LAYOUT_UNSUPPORTED = "layout_unsupported"  # Non-palettized or unsupported layout


@dataclass
class ReplacementDiag:
    """Per-replacement diagnostic record."""
    base_name: str
    reason: ReplacementReason
    tensor_class: str = ""          # e.g. "palettized_lut", "fp16_const"
    bytes_saved: int = 0            # Estimated bytes saved (idx + lut)
    cos_sim: float = 0.0
    max_abs_diff: float = 0.0
    mean_abs_diff: float = 0.0
    anchor_shape: Tuple = ()
    target_shape: Tuple = ()


@dataclass
class DeduplicationReport:
    """Summary report of deduplication operation."""
    total_pairs_evaluated: int = 0
    pairs_replaced: int = 0
    pairs_identical: int = 0
    pairs_skipped: int = 0
    bytes_saved: int = 0
    diagnostics: List[ReplacementDiag] = None

    def __post_init__(self):
        if self.diagnostics is None:
            self.diagnostics = []

    def summary(self) -> str:
        """Return human-readable summary."""
        return (
            f"Deduplication Report:\n"
            f"  Total pairs evaluated: {self.total_pairs_evaluated}\n"
            f"  Pairs replaced: {self.pairs_replaced}\n"
            f"  Pairs already identical: {self.pairs_identical}\n"
            f"  Pairs skipped: {self.pairs_skipped}\n"
            f"  Estimated bytes saved: {self.bytes_saved / 1e6:.2f} MB"
        )


# ---------------------------------------------------------------------------
# Tensor class classification
# ---------------------------------------------------------------------------

def _classify_tensor(nk: str) -> str:
    """Classify a normalized key into a tensor class for matching."""
    if "_palettized_indices" in nk:
        return "palettized_indices"
    if "_palettized_lut" in nk:
        return "palettized_lut"
    if "_weight" in nk and "_palettized" not in nk:
        return "dense_weight"
    if "_bias" in nk:
        return "bias"
    if "causal_mask" in nk or "mask" in nk:
        return "mask"
    if "kv_cache" in nk or "cache" in nk:
        return "cache"
    return "other"


def _is_palettized_pair_key(nk: str) -> bool:
    """Check if a key is part of a palettized weight pair."""
    return "_palettized_indices" in nk or "_palettized_lut" in nk


# ---------------------------------------------------------------------------
# Preflight compatibility checks
# ---------------------------------------------------------------------------

class PreflightError(Exception):
    """Raised when preflight compatibility checks fail."""
    pass


def _preflight_check_io_signature(anchor_prog, target_prog,
                                  anchor_fn: str = "main",
                                  target_fn: str = "main"):
    """Verify anchor and target have compatible I/O signatures."""
    a_func = anchor_prog.functions.get(anchor_fn)
    t_func = target_prog.functions.get(target_fn)

    if a_func is None:
        raise PreflightError(f"Anchor has no function '{anchor_fn}'")
    if t_func is None:
        raise PreflightError(f"Target has no function '{target_fn}'")

    # Compare input names
    a_inputs = {inp.name: tuple(inp.shape) for inp in a_func.inputs.values()
                if hasattr(inp, 'shape')}
    t_inputs = {inp.name: tuple(inp.shape) for inp in t_func.inputs.values()
                if hasattr(inp, 'shape')}

    a_names = set(a_inputs.keys())
    t_names = set(t_inputs.keys())

    if a_names != t_names:
        raise PreflightError(
            f"I/O signature mismatch: input names differ. "
            f"Anchor-only: {a_names - t_names}, Target-only: {t_names - a_names}"
        )


def _preflight_check_weight_counts(anchor_weights, target_weights):
    """Verify anchor and target have similar weight tensor counts."""
    a_count = len(anchor_weights)
    t_count = len(target_weights)

    if a_count == 0 or t_count == 0:
        raise PreflightError(
            f"Empty weight set: anchor has {a_count}, target has {t_count} tensors"
        )

    ratio = min(a_count, t_count) / max(a_count, t_count)
    if ratio < 0.8:
        raise PreflightError(
            f"Weight count mismatch: anchor has {a_count}, target has {t_count} tensors "
            f"(ratio {ratio:.2f} < 0.8 threshold)."
        )


def _preflight_check_palettized_config(anchor_weights, target_weights):
    """Verify LUT configurations are compatible."""
    def _get_lut_configs(weights):
        configs = {}
        for nk, arr in weights.items():
            if "_palettized_lut" in nk:
                base = nk.replace("_palettized_lut", "")
                squeezed = arr.squeeze()
                if squeezed.ndim == 2:
                    configs[base] = (squeezed.shape[0], squeezed.shape[1])
                elif squeezed.ndim == 1:
                    configs[base] = (1, squeezed.shape[0])
        return configs

    a_configs = _get_lut_configs(anchor_weights)
    t_configs = _get_lut_configs(target_weights)

    mismatched = []
    for base in a_configs:
        if base in t_configs and a_configs[base] != t_configs[base]:
            mismatched.append(
                f"  {base}: anchor={a_configs[base]} vs target={t_configs[base]}"
            )

    if mismatched:
        raise PreflightError(
            f"LUT configuration mismatch for {len(mismatched)} weight(s):\n"
            + "\n".join(mismatched[:5])
        )


# ---------------------------------------------------------------------------
# MIL weight extraction
# ---------------------------------------------------------------------------

def _load_mil_program(mlpackage_path: str):
    """Load an mlpackage into a MIL program with weight data resolved."""
    import coremltools as ct
    from coremltools.converters.mil.frontend.milproto.load import load as mil_load

    model = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = model.get_spec()
    prog = mil_load(
        spec,
        specification_version=spec.specificationVersion,
        file_weights_dir=model.weights_dir,
    )
    return prog, model


def _normalize_key(name: str) -> str:
    """Strip trailing auto-generated _N suffix for cross-trace matching."""
    return re.sub(r"_(\d+)$", "", name)


def _extract_const_weights(prog, func_name: str = "main") -> Dict[str, np.ndarray]:
    """Extract {normalized_key: numpy_array} for all non-scalar const ops."""
    weights = {}
    func = prog.functions.get(func_name)
    if func is None:
        func = next(iter(prog.functions.values()), None)
    if func is None:
        return weights

    for op in func.find_ops(op_type="const"):
        val = op.val
        if val is None:
            continue
        arr = val.val if hasattr(val, "val") else val
        if isinstance(arr, np.ndarray) and arr.size > 1:
            nk = _normalize_key(op.name)
            weights[nk] = arr
    return weights


# ---------------------------------------------------------------------------
# Dequantization for verification
# ---------------------------------------------------------------------------

def _dequantize_lut(indices: np.ndarray, lut: np.ndarray) -> Optional[np.ndarray]:
    """Dequantize palettized weight: value = lut[group, index].

    Handles CoreML's multi-dimensional tensor layouts:
      - indices: (O, I, 1, 1) or (O, I) — uint8 bin assignments
      - lut: (G, 1, 1, 1, C, 1) or (G, C) — fp16 palette per group

    Returns:
        Dequantized array with same number of elements as indices, or None on failure.
    """
    idx = indices.squeeze()
    lt = lut.squeeze()

    if idx.ndim == 1:
        idx = idx.reshape(-1, 1)
    if idx.ndim != 2:
        return None

    if lt.ndim == 1:
        lt = lt.reshape(1, -1)
    if lt.ndim != 2:
        return None

    O, I = idx.shape
    G, C = lt.shape
    if G == 0 or O == 0 or C == 0:
        return None
    group_size = max(1, O // G)

    group_ids = np.minimum(np.arange(O) // group_size, G - 1)
    result = lt[group_ids[:, None], idx]
    return result


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two flattened arrays."""
    a_flat = a.astype(np.float64).ravel()
    b_flat = b.astype(np.float64).ravel()
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-30 or norm_b < 1e-30:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Core: find and verify replaceable weight pairs
# ---------------------------------------------------------------------------

def find_replaceable_weights(
    anchor_weights: Dict[str, np.ndarray],
    target_weights: Dict[str, np.ndarray],
    cos_threshold: float = 0.9999,
    max_abs_threshold: Optional[float] = None,
    mean_abs_threshold: float = 0.001,
    verify_dequant: bool = True,
    verbose: bool = False,
    diagnostics: Optional[List[ReplacementDiag]] = None,
) -> Dict[str, str]:
    """Find target weight keys that can be safely replaced with anchor values.

    Matches palettized weight pairs (_palettized_indices + _palettized_lut)
    and verifies that dequantized values are semantically identical.

    Args:
        anchor_weights: Extracted weights from the anchor model
        target_weights: Extracted weights from the target model
        cos_threshold: Minimum cosine similarity (default 0.9999)
        max_abs_threshold: Maximum absolute difference (None=relaxed)
        mean_abs_threshold: Maximum mean absolute difference (default 0.001)
        verify_dequant: If True, verify via dequantization before replacing
        verbose: Print per-weight replacement details
        diagnostics: If provided, append ReplacementDiag records

    Returns:
        Dict mapping target normalized_key -> anchor normalized_key
    """
    replacements: Dict[str, str] = {}

    def _base_name(nk: str) -> Optional[str]:
        tc = _classify_tensor(nk)
        if tc == "palettized_indices":
            return nk.replace("_palettized_indices", "")
        if tc == "palettized_lut":
            return nk.replace("_palettized_lut", "")
        return None

    # Find all palettized weight bases in anchor
    anchor_bases = set()
    for nk in anchor_weights:
        base = _base_name(nk)
        if base is not None:
            anchor_bases.add(base)

    replaced_count = 0
    skipped_count = 0
    already_identical = 0
    total_bytes_saved = 0

    for base in sorted(anchor_bases):
        idx_key = f"{base}_palettized_indices"
        lut_key = f"{base}_palettized_lut"

        if idx_key not in anchor_weights or lut_key not in anchor_weights:
            continue
        if idx_key not in target_weights or lut_key not in target_weights:
            continue

        a_idx = anchor_weights[idx_key]
        a_lut = anchor_weights[lut_key]
        t_idx = target_weights[idx_key]
        t_lut = target_weights[lut_key]

        # Shape must match
        if a_idx.shape != t_idx.shape or a_lut.shape != t_lut.shape:
            if verbose:
                print(f"    SKIP (shape): {base}")
            skipped_count += 1
            if diagnostics is not None:
                diagnostics.append(ReplacementDiag(
                    base_name=base,
                    reason=ReplacementReason.REJECTED_SHAPE,
                    tensor_class="palettized",
                    anchor_shape=a_idx.shape,
                    target_shape=t_idx.shape,
                ))
            continue

        # Already identical?
        if np.array_equal(a_idx, t_idx) and np.array_equal(a_lut, t_lut):
            already_identical += 1
            if diagnostics is not None:
                diagnostics.append(ReplacementDiag(
                    base_name=base,
                    reason=ReplacementReason.IDENTICAL,
                    tensor_class="palettized",
                    anchor_shape=a_idx.shape,
                    target_shape=t_idx.shape,
                ))
            continue

        pair_bytes = a_idx.nbytes + a_lut.nbytes

        if verify_dequant:
            a_deq = _dequantize_lut(a_idx, a_lut)
            t_deq = _dequantize_lut(t_idx, t_lut)

            if a_deq is None or t_deq is None:
                if verbose:
                    print(f"    SKIP (deq failed): {base}")
                skipped_count += 1
                if diagnostics is not None:
                    diagnostics.append(ReplacementDiag(
                        base_name=base,
                        reason=ReplacementReason.REJECTED_DEQ_FAIL,
                        tensor_class="palettized",
                        anchor_shape=a_idx.shape,
                        target_shape=t_idx.shape,
                    ))
                continue

            cos = _cosine_similarity(a_deq, t_deq)
            diff = np.abs(a_deq.astype(np.float64) - t_deq.astype(np.float64))
            max_abs = float(np.max(diff))
            mean_abs = float(np.mean(diff))

            if max_abs_threshold is None:
                accepted = (cos >= cos_threshold and mean_abs <= mean_abs_threshold)
            else:
                accepted = (cos >= cos_threshold
                            and max_abs <= max_abs_threshold
                            and mean_abs <= mean_abs_threshold)

            if not accepted:
                if verbose:
                    print(f"    SKIP (threshold): {base} cos={cos:.6f}")
                skipped_count += 1
                if diagnostics is not None:
                    diagnostics.append(ReplacementDiag(
                        base_name=base,
                        reason=ReplacementReason.REJECTED_THRESHOLD,
                        tensor_class="palettized",
                        cos_sim=cos,
                        max_abs_diff=max_abs,
                        mean_abs_diff=mean_abs,
                        anchor_shape=a_idx.shape,
                        target_shape=t_idx.shape,
                    ))
                continue

            if verbose:
                print(f"    REPLACE (cos={cos:.6f}, ~{pair_bytes/1024:.0f}KB): {base}")

            if diagnostics is not None:
                diagnostics.append(ReplacementDiag(
                    base_name=base,
                    reason=ReplacementReason.DEQ_CLOSE,
                    tensor_class="palettized",
                    bytes_saved=pair_bytes,
                    cos_sim=cos,
                    max_abs_diff=max_abs,
                    mean_abs_diff=mean_abs,
                    anchor_shape=a_idx.shape,
                    target_shape=t_idx.shape,
                ))
        else:
            if verbose:
                print(f"    REPLACE (no verify, ~{pair_bytes/1024:.0f}KB): {base}")
            if diagnostics is not None:
                diagnostics.append(ReplacementDiag(
                    base_name=base,
                    reason=ReplacementReason.REPLACED_NO_VERIFY,
                    tensor_class="palettized",
                    bytes_saved=pair_bytes,
                    anchor_shape=a_idx.shape,
                    target_shape=t_idx.shape,
                ))

        replacements[idx_key] = idx_key
        replacements[lut_key] = lut_key
        replaced_count += 1
        total_bytes_saved += pair_bytes

    if verbose or replaced_count > 0:
        print(f"  Dedup summary: {replaced_count} weight pairs to replace "
              f"(~{total_bytes_saved / 1e6:.1f} MB), "
              f"{already_identical} already identical, {skipped_count} skipped")

    return replacements


# ---------------------------------------------------------------------------
# Apply replacements to MIL program
# ---------------------------------------------------------------------------

def _apply_replacements_to_mlpackage(
    source_path: str,
    anchor_weights: Dict[str, np.ndarray],
    replacements: Dict[str, str],
    output_path: str,
    src_func_name: str = "main",
    verbose: bool = False,
) -> int:
    """Load source mlpackage, replace specified const ops, save to output_path.

    Returns number of ops replaced.
    """
    import coremltools as ct
    from coremltools.converters.mil.frontend.milproto.load import load as mil_load

    model = ct.models.MLModel(source_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = model.get_spec()
    prog = mil_load(
        spec,
        specification_version=spec.specificationVersion,
        file_weights_dir=model.weights_dir,
    )

    func = prog.functions.get(src_func_name)
    if func is None:
        func = next(iter(prog.functions.values()), None)
    if func is None:
        print(f"  WARNING: no function found in {source_path}")
        return 0

    replaced = 0
    for op in func.find_ops(op_type="const"):
        nk = _normalize_key(op.name)
        if nk not in replacements:
            continue
        anchor_nk = replacements[nk]
        if anchor_nk not in anchor_weights:
            continue

        anchor_arr = anchor_weights[anchor_nk]
        out_var = op.outputs[0]
        current_arr = out_var.val
        if current_arr is None or not isinstance(current_arr, np.ndarray):
            continue
        if current_arr.shape != anchor_arr.shape:
            continue
        if np.array_equal(current_arr, anchor_arr):
            continue

        out_var._sym_val.val = anchor_arr.copy().reshape(current_arr.shape)
        replaced += 1

    if replaced > 0:
        from coremltools.converters.mil.converter import mil_convert as _mil_convert

        if prog.default_function_name not in prog.functions:
            prog.default_function_name = src_func_name

        _spec_to_target = {
            7: ct.target.iOS16,
            8: ct.target.iOS17,
            9: ct.target.iOS18,
        }
        deploy_target = _spec_to_target.get(spec.specificationVersion, ct.target.iOS18)

        mlmodel = _mil_convert(
            prog,
            convert_from="milinternal",
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.CPU_ONLY,
            skip_model_load=True,
            pass_pipeline=ct.PassPipeline.EMPTY,
            minimum_deployment_target=deploy_target,
        )
        mlmodel.save(output_path)

        if verbose:
            print(f"    Saved modified model ({replaced} ops replaced) -> {output_path}")
    else:
        if source_path != output_path:
            shutil.copytree(source_path, output_path)
        if verbose:
            print(f"    No replacements needed, copied as-is -> {output_path}")

    return replaced


# ---------------------------------------------------------------------------
# High-level API: prepare deduped sources for save_multifunction
# ---------------------------------------------------------------------------

@contextmanager
def prepare_dedup_sources(
    sources: List[Tuple[str, str, str]],
    cos_threshold: float = 0.9999,
    max_abs_threshold: Optional[float] = None,
    mean_abs_threshold: float = 0.001,
    verify_dequant: bool = True,
    verbose: bool = False,
    temp_dir: Optional[str] = None,
    preflight: bool = True,
    diagnostics: Optional[List[ReplacementDiag]] = None,
):
    """Context manager that prepares dedup-optimized source mlpackages.

    Takes a list of (mlpackage_path, src_function_name, target_function_name) tuples.
    The first entry is treated as the anchor. For all subsequent entries, palettized
    weights that are semantically identical to the anchor's are replaced.

    Args:
        sources: List of (mlpackage_path, src_function_name, target_function_name)
        cos_threshold: Minimum cosine similarity (default 0.9999)
        max_abs_threshold: Maximum absolute difference (None=relaxed)
        mean_abs_threshold: Maximum mean absolute difference (default 0.001)
        verify_dequant: If True, verify via dequantization before replacing
        verbose: Print per-weight replacement details
        temp_dir: Override temp directory
        preflight: If True, run compatibility checks before dedup
        diagnostics: If provided, append per-replacement diagnostic records

    Yields:
        List of (path, src_fn, tgt_fn) ready for MultiFunctionDescriptor
    """
    if len(sources) < 2:
        yield sources
        return

    tmp_root = tempfile.mkdtemp(prefix="nimbo_dedup_", dir=temp_dir)
    result = []
    total_replaced = 0

    try:
        t0 = time.time()

        # Step 1: Extract anchor weights
        anchor_path, anchor_src_fn, anchor_tgt_fn = sources[0]
        print(f"[nimbo-dedup] Loading anchor: {os.path.basename(anchor_path)}")
        anchor_prog, _ = _load_mil_program(anchor_path)
        anchor_weights = _extract_const_weights(anchor_prog, anchor_src_fn)
        print(f"[nimbo-dedup] Anchor has {len(anchor_weights)} weight tensors")

        result.append((anchor_path, anchor_src_fn, anchor_tgt_fn))

        # Step 2: Process each non-anchor source
        for i, (src_path, src_fn, tgt_fn) in enumerate(sources[1:], 1):
            print(f"[nimbo-dedup] Processing target {i}/{len(sources)-1}: "
                  f"{os.path.basename(src_path)} -> {tgt_fn}")

            target_prog, _ = _load_mil_program(src_path)
            target_weights = _extract_const_weights(target_prog, src_fn)

            if preflight:
                try:
                    _preflight_check_io_signature(
                        anchor_prog, target_prog, anchor_src_fn, src_fn)
                    _preflight_check_weight_counts(anchor_weights, target_weights)
                    _preflight_check_palettized_config(anchor_weights, target_weights)
                    if verbose:
                        print(f"[nimbo-dedup]   Preflight checks passed")
                except PreflightError as e:
                    print(f"[nimbo-dedup]   WARNING: Preflight failed: {e}")
                    result.append((src_path, src_fn, tgt_fn))
                    continue

            replacements = find_replaceable_weights(
                anchor_weights, target_weights,
                cos_threshold=cos_threshold,
                max_abs_threshold=max_abs_threshold,
                mean_abs_threshold=mean_abs_threshold,
                verify_dequant=verify_dequant,
                verbose=verbose,
                diagnostics=diagnostics,
            )

            if not replacements:
                print(f"[nimbo-dedup]   No replacements needed")
                result.append((src_path, src_fn, tgt_fn))
                continue

            temp_pkg = os.path.join(tmp_root, f"dedup_{i}_{os.path.basename(src_path)}")
            n_replaced = _apply_replacements_to_mlpackage(
                src_path, anchor_weights, replacements, temp_pkg,
                src_func_name=src_fn, verbose=verbose,
            )
            total_replaced += n_replaced

            if n_replaced > 0:
                result.append((temp_pkg, src_fn, tgt_fn))
                print(f"[nimbo-dedup]   Replaced {n_replaced} const ops")
            else:
                result.append((src_path, src_fn, tgt_fn))

        elapsed = time.time() - t0
        print(f"[nimbo-dedup] Done in {elapsed:.1f}s: {total_replaced} total ops replaced")

        if diagnostics is not None and len(diagnostics) > 0:
            _print_diagnostics_summary(diagnostics)

        yield result

    finally:
        if os.path.exists(tmp_root):
            shutil.rmtree(tmp_root, ignore_errors=True)


def _print_diagnostics_summary(diagnostics: List[ReplacementDiag]):
    """Print a summary table of per-replacement diagnostics."""
    by_reason = {}
    for d in diagnostics:
        by_reason.setdefault(d.reason.value, []).append(d)

    total_saved = sum(d.bytes_saved for d in diagnostics)

    print(f"\n[nimbo-dedup] Diagnostics ({len(diagnostics)} weight pairs evaluated):")
    for reason, items in sorted(by_reason.items()):
        saved = sum(d.bytes_saved for d in items)
        print(f"  {reason}: {len(items)} pairs"
              + (f" (~{saved / 1e6:.1f} MB saved)" if saved > 0 else ""))

    if total_saved > 0:
        print(f"  Total estimated savings: ~{total_saved / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# Verification: ensure deduplication preserves model outputs
# ---------------------------------------------------------------------------

def verify_dedup_correctness(
    original_path: str,
    deduped_path: str,
    num_samples: int = 5,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    verbose: bool = True,
) -> Tuple[bool, Dict]:
    """Verify that a deduped model produces the same outputs as the original.

    Runs random inputs through both models and compares outputs.

    Args:
        original_path: Path to original .mlpackage
        deduped_path: Path to deduped .mlpackage
        num_samples: Number of random inputs to test
        rtol: Relative tolerance for np.allclose
        atol: Absolute tolerance for np.allclose
        verbose: Print progress and results

    Returns:
        Tuple of (is_correct: bool, report: dict)
    """
    import coremltools as ct

    if verbose:
        print(f"\n[nimbo-dedup] Verifying deduplication correctness...")
        print(f"  Original: {original_path}")
        print(f"  Deduped:  {deduped_path}")

    # Load both models
    original_model = ct.models.MLModel(original_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    deduped_model = ct.models.MLModel(deduped_path, compute_units=ct.ComputeUnit.CPU_ONLY)

    # Get input specs
    original_spec = original_model.get_spec()
    inputs = original_spec.description.input

    report = {
        "num_samples": num_samples,
        "inputs_tested": [],
        "outputs_compared": [],
        "max_abs_diff": 0.0,
        "max_rel_diff": 0.0,
        "all_close": True,
        "mismatches": [],
    }

    for sample_idx in range(num_samples):
        if verbose:
            print(f"  Testing sample {sample_idx + 1}/{num_samples}...")

        # Generate random inputs based on model spec
        input_dict = {}
        for inp in inputs:
            name = inp.name
            if inp.type.HasField("multiArrayType"):
                arr_type = inp.type.multiArrayType
                shape = [d for d in arr_type.shape]
                if arr_type.dataType == 65568:  # Float16
                    data = np.random.randn(*shape).astype(np.float16)
                else:  # Float32
                    data = np.random.randn(*shape).astype(np.float32)
                input_dict[name] = data
            elif inp.type.HasField("imageType"):
                # Skip image inputs for now
                continue

        if not input_dict:
            if verbose:
                print("  WARNING: No valid inputs found, skipping verification")
            return True, report

        report["inputs_tested"].append(list(input_dict.keys()))

        # Run both models
        try:
            original_output = original_model.predict(input_dict)
            deduped_output = deduped_model.predict(input_dict)
        except Exception as e:
            if verbose:
                print(f"  ERROR during inference: {e}")
            report["mismatches"].append(f"Inference error: {e}")
            report["all_close"] = False
            continue

        # Compare outputs
        for key in original_output:
            if key not in deduped_output:
                report["mismatches"].append(f"Missing output: {key}")
                report["all_close"] = False
                continue

            orig_val = np.array(original_output[key])
            dedup_val = np.array(deduped_output[key])

            if orig_val.shape != dedup_val.shape:
                report["mismatches"].append(f"Shape mismatch for {key}: {orig_val.shape} vs {dedup_val.shape}")
                report["all_close"] = False
                continue

            # Calculate differences
            abs_diff = np.abs(orig_val.astype(np.float64) - dedup_val.astype(np.float64))
            max_abs = float(np.max(abs_diff))
            mean_abs = float(np.mean(abs_diff))

            # Relative difference (avoid division by zero)
            denom = np.maximum(np.abs(orig_val.astype(np.float64)), 1e-10)
            rel_diff = abs_diff / denom
            max_rel = float(np.max(rel_diff))

            report["max_abs_diff"] = max(report["max_abs_diff"], max_abs)
            report["max_rel_diff"] = max(report["max_rel_diff"], max_rel)

            is_close = np.allclose(orig_val, dedup_val, rtol=rtol, atol=atol)
            report["outputs_compared"].append({
                "key": key,
                "sample": sample_idx,
                "max_abs_diff": max_abs,
                "mean_abs_diff": mean_abs,
                "max_rel_diff": max_rel,
                "is_close": is_close,
            })

            if not is_close:
                report["all_close"] = False
                report["mismatches"].append(
                    f"Output '{key}' sample {sample_idx}: max_abs={max_abs:.2e}, max_rel={max_rel:.2e}"
                )

    is_correct = report["all_close"]

    if verbose:
        print(f"\n[nimbo-dedup] Verification {'PASSED' if is_correct else 'FAILED'}:")
        print(f"  Max absolute difference: {report['max_abs_diff']:.2e}")
        print(f"  Max relative difference: {report['max_rel_diff']:.2e}")
        if report["mismatches"]:
            print(f"  Mismatches ({len(report['mismatches'])}):")
            for m in report["mismatches"][:5]:
                print(f"    - {m}")

    return is_correct, report


def verify_weight_equivalence(
    anchor_path: str,
    target_path: str,
    verbose: bool = True,
) -> Tuple[bool, Dict]:
    """Verify that two models have semantically equivalent weights.

    Compares dequantized weight values between anchor and target models.

    Args:
        anchor_path: Path to anchor .mlpackage
        target_path: Path to target .mlpackage
        verbose: Print progress and results

    Returns:
        Tuple of (is_equivalent: bool, report: dict)
    """
    if verbose:
        print(f"\n[nimbo-dedup] Verifying weight equivalence...")
        print(f"  Anchor: {anchor_path}")
        print(f"  Target: {target_path}")

    # Load programs
    anchor_prog, _ = _load_mil_program(anchor_path)
    target_prog, _ = _load_mil_program(target_path)

    anchor_weights = _extract_const_weights(anchor_prog)
    target_weights = _extract_const_weights(target_prog)

    report = {
        "anchor_count": len(anchor_weights),
        "target_count": len(target_weights),
        "pairs_compared": 0,
        "identical_pairs": 0,
        "equivalent_pairs": 0,
        "different_pairs": 0,
        "max_cos_diff": 0.0,
        "differences": [],
    }

    # Find palettized weight bases in both
    def get_bases(weights):
        bases = set()
        for nk in weights:
            if "_palettized_indices" in nk:
                bases.add(nk.replace("_palettized_indices", ""))
        return bases

    anchor_bases = get_bases(anchor_weights)
    target_bases = get_bases(target_weights)
    common_bases = anchor_bases & target_bases

    if verbose:
        print(f"  Anchor bases: {len(anchor_bases)}")
        print(f"  Target bases: {len(target_bases)}")
        print(f"  Common bases: {len(common_bases)}")

    for base in sorted(common_bases):
        idx_key = f"{base}_palettized_indices"
        lut_key = f"{base}_palettized_lut"

        if idx_key not in anchor_weights or lut_key not in anchor_weights:
            continue
        if idx_key not in target_weights or lut_key not in target_weights:
            continue

        a_idx = anchor_weights[idx_key]
        a_lut = anchor_weights[lut_key]
        t_idx = target_weights[idx_key]
        t_lut = target_weights[lut_key]

        report["pairs_compared"] += 1

        # Byte-identical?
        if np.array_equal(a_idx, t_idx) and np.array_equal(a_lut, t_lut):
            report["identical_pairs"] += 1
            continue

        # Dequantize and compare
        a_deq = _dequantize_lut(a_idx, a_lut)
        t_deq = _dequantize_lut(t_idx, t_lut)

        if a_deq is None or t_deq is None:
            report["different_pairs"] += 1
            report["differences"].append(f"{base}: dequantization failed")
            continue

        cos = _cosine_similarity(a_deq, t_deq)
        report["max_cos_diff"] = max(report["max_cos_diff"], 1.0 - cos)

        if cos >= 0.9999:
            report["equivalent_pairs"] += 1
        else:
            report["different_pairs"] += 1
            report["differences"].append(f"{base}: cos={cos:.6f}")

    is_equivalent = (report["different_pairs"] == 0)

    if verbose:
        print(f"\n[nimbo-dedup] Weight equivalence {'PASSED' if is_equivalent else 'FAILED'}:")
        print(f"  Pairs compared: {report['pairs_compared']}")
        print(f"  Identical: {report['identical_pairs']}")
        print(f"  Equivalent (cos>=0.9999): {report['equivalent_pairs']}")
        print(f"  Different: {report['different_pairs']}")
        if report["differences"]:
            print(f"  Differences ({len(report['differences'])}):")
            for d in report["differences"][:5]:
                print(f"    - {d}")

    return is_equivalent, report
