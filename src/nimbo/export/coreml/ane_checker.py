# Copyright (c) 2025, Nimbo
# Licensed under the Apache License, Version 2.0

"""
Apple Neural Engine (ANE) Compatibility Checker

This module analyzes PyTorch models and CoreML models to determine which layers
can run efficiently on the Apple Neural Engine (ANE) and identifies potential
compatibility issues.

ANE Compatibility Rules:
1. Shape constraints: ANE prefers dimensions divisible by 4, 8, or 16
2. Operation support: Some operations fall back to CPU/GPU
3. Memory limits: Large tensors may not fit in ANE memory
4. Data types: ANE works best with float16

Usage:
    from nimbo.export.coreml.ane_checker import ANEChecker

    # Check PyTorch model
    checker = ANEChecker()
    report = checker.check_pytorch_model(model, input_shape=(1, 512))
    checker.save_report(report, "ane_report.txt")

    # Check CoreML model
    report = checker.check_coreml_model("model.mlpackage")
    checker.save_report(report, "ane_report.txt")
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime

import torch
import torch.nn as nn


class ANEStatus(Enum):
    """ANE compatibility status for a layer."""
    COMPATIBLE = "compatible"           # Fully ANE compatible
    OPTIMIZED = "optimized"             # Already optimized for ANE (e.g., Conv2d)
    NEEDS_CONVERSION = "needs_conversion"  # Needs conversion (e.g., Linear -> Conv2d)
    PARTIAL = "partial"                 # Partially compatible, may fall back
    INCOMPATIBLE = "incompatible"       # Will run on CPU/GPU
    UNKNOWN = "unknown"                 # Cannot determine


class ANEIssueLevel(Enum):
    """Severity level for ANE compatibility issues."""
    INFO = "info"           # Informational
    WARNING = "warning"     # May impact performance
    ERROR = "error"         # Will not run on ANE


@dataclass
class ANEIssue:
    """A single ANE compatibility issue."""
    level: ANEIssueLevel
    layer_name: str
    layer_type: str
    message: str
    suggestion: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ANELayerReport:
    """Report for a single layer."""
    name: str
    type: str
    status: ANEStatus
    input_shape: Tuple = ()
    output_shape: Tuple = ()
    param_count: int = 0
    issues: List[ANEIssue] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class ANEReport:
    """Complete ANE compatibility report."""
    model_name: str
    timestamp: str
    total_layers: int = 0
    compatible_layers: int = 0
    optimized_layers: int = 0
    needs_conversion_layers: int = 0
    partial_layers: int = 0
    incompatible_layers: int = 0
    layers: List[ANELayerReport] = field(default_factory=list)
    issues: List[ANEIssue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def get_compatibility_score(self) -> float:
        """Calculate overall ANE compatibility score (0-100)."""
        if self.total_layers == 0:
            return 0.0

        # Weights for different statuses
        weights = {
            ANEStatus.OPTIMIZED: 1.0,
            ANEStatus.COMPATIBLE: 1.0,
            ANEStatus.NEEDS_CONVERSION: 0.7,  # Can be fixed
            ANEStatus.PARTIAL: 0.5,
            ANEStatus.INCOMPATIBLE: 0.0,
            ANEStatus.UNKNOWN: 0.3,
        }

        total_score = sum(
            weights.get(layer.status, 0.0) for layer in self.layers
        )

        return (total_score / self.total_layers) * 100


# =============================================================================
# ANE Compatibility Rules
# =============================================================================

# Dimensions that ANE handles efficiently
ANE_OPTIMAL_DIMS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Maximum tensor size for ANE (approximate, varies by device)
ANE_MAX_TENSOR_SIZE = 16 * 1024 * 1024  # 16M elements

# Operations fully supported by ANE
ANE_SUPPORTED_OPS = {
    "conv2d", "conv1d", "depthwise_conv2d",
    "relu", "relu6", "gelu", "silu", "swish", "sigmoid", "tanh", "softmax",
    "add", "mul", "sub", "div",
    "matmul", "bmm",
    "layer_norm", "batch_norm", "instance_norm", "group_norm",
    "max_pool2d", "avg_pool2d", "adaptive_avg_pool2d",
    "concat", "split", "reshape", "transpose", "permute",
    "upsample", "interpolate",
}

# Operations that may fall back to CPU/GPU
ANE_PARTIAL_OPS = {
    "linear",  # Works but Conv2d is faster
    "embedding",  # Large embeddings may fall back
    "gather", "scatter",
    "topk", "sort", "argsort",
    "einsum",
    "custom_op",
}

# Operations not supported on ANE
ANE_UNSUPPORTED_OPS = {
    "lstm", "gru", "rnn",  # Recurrent layers
    "fft", "ifft",
    "conv3d", "conv_transpose3d",
    "grid_sample",
    "roi_pool", "roi_align",
    "nms",
}


class ANEChecker:
    """Check model compatibility with Apple Neural Engine."""

    def __init__(self, verbose: bool = False):
        """Initialize ANE checker.

        Args:
            verbose: Print detailed information during checking
        """
        self.verbose = verbose
        self._layer_count = 0

    def check_pytorch_model(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = None,
        model_name: str = None,
    ) -> ANEReport:
        """Check a PyTorch model for ANE compatibility.

        Args:
            model: PyTorch model to check
            input_shape: Input shape for the model (batch, seq_len) or (batch, channels, height, width)
            model_name: Name for the report

        Returns:
            ANEReport with compatibility analysis
        """
        self._layer_count = 0
        model_name = model_name or model.__class__.__name__

        report = ANEReport(
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
        )

        if self.verbose:
            print(f"Checking PyTorch model: {model_name}")

        # Analyze each layer
        for name, module in model.named_modules():
            if len(list(module.children())) > 0:
                # Skip container modules
                continue

            layer_report = self._check_pytorch_layer(name, module)
            report.layers.append(layer_report)
            report.total_layers += 1

            # Update counters
            if layer_report.status == ANEStatus.COMPATIBLE:
                report.compatible_layers += 1
            elif layer_report.status == ANEStatus.OPTIMIZED:
                report.optimized_layers += 1
            elif layer_report.status == ANEStatus.NEEDS_CONVERSION:
                report.needs_conversion_layers += 1
            elif layer_report.status == ANEStatus.PARTIAL:
                report.partial_layers += 1
            elif layer_report.status == ANEStatus.INCOMPATIBLE:
                report.incompatible_layers += 1

            # Collect issues
            report.issues.extend(layer_report.issues)

        # Generate summary
        report.summary = self._generate_summary(report)

        return report

    def _check_pytorch_layer(self, name: str, module: nn.Module) -> ANELayerReport:
        """Check a single PyTorch layer for ANE compatibility."""
        layer_type = module.__class__.__name__
        param_count = sum(p.numel() for p in module.parameters())

        layer_report = ANELayerReport(
            name=name,
            type=layer_type,
            status=ANEStatus.UNKNOWN,
            param_count=param_count,
        )

        # Check by layer type
        if isinstance(module, nn.Linear):
            self._check_linear_layer(module, layer_report)

        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            self._check_conv_layer(module, layer_report)

        elif isinstance(module, nn.LayerNorm):
            self._check_layernorm(module, layer_report)

        elif isinstance(module, nn.Embedding):
            self._check_embedding(module, layer_report)

        elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh)):
            layer_report.status = ANEStatus.COMPATIBLE
            layer_report.notes.append("Activation function is ANE compatible")

        elif isinstance(module, nn.Softmax):
            self._check_softmax(module, layer_report)

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            layer_report.status = ANEStatus.COMPATIBLE
            layer_report.notes.append("BatchNorm is ANE compatible")

        elif isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.AvgPool1d, nn.AvgPool2d)):
            layer_report.status = ANEStatus.COMPATIBLE
            layer_report.notes.append("Pooling operation is ANE compatible")

        elif isinstance(module, nn.Dropout):
            layer_report.status = ANEStatus.COMPATIBLE
            layer_report.notes.append("Dropout is removed in inference mode")

        elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
            self._check_recurrent_layer(module, layer_report)

        elif isinstance(module, nn.MultiheadAttention):
            self._check_attention(module, layer_report)

        else:
            # Unknown layer type
            layer_report.status = ANEStatus.UNKNOWN
            layer_report.notes.append(f"Unknown layer type: {layer_type}")

        return layer_report

    def _check_linear_layer(self, module: nn.Linear, report: ANELayerReport):
        """Check Linear layer for ANE compatibility."""
        in_features = module.in_features
        out_features = module.out_features

        report.input_shape = (in_features,)
        report.output_shape = (out_features,)

        # Linear works on ANE but Conv2d is more efficient
        report.status = ANEStatus.NEEDS_CONVERSION

        # Check dimension alignment
        issues = []

        if in_features % 4 != 0:
            issues.append(ANEIssue(
                level=ANEIssueLevel.WARNING,
                layer_name=report.name,
                layer_type="Linear",
                message=f"Input features ({in_features}) not divisible by 4",
                suggestion="Consider padding to nearest multiple of 4 for better ANE efficiency",
                details={"in_features": in_features, "recommended": ((in_features + 3) // 4) * 4},
            ))

        if out_features % 4 != 0:
            issues.append(ANEIssue(
                level=ANEIssueLevel.WARNING,
                layer_name=report.name,
                layer_type="Linear",
                message=f"Output features ({out_features}) not divisible by 4",
                suggestion="Consider padding to nearest multiple of 4 for better ANE efficiency",
                details={"out_features": out_features, "recommended": ((out_features + 3) // 4) * 4},
            ))

        # Check for very large layers
        total_params = in_features * out_features
        if total_params > ANE_MAX_TENSOR_SIZE:
            issues.append(ANEIssue(
                level=ANEIssueLevel.ERROR,
                layer_name=report.name,
                layer_type="Linear",
                message=f"Layer too large for ANE ({total_params:,} params)",
                suggestion="Consider splitting into smaller chunks",
                details={"param_count": total_params, "max_recommended": ANE_MAX_TENSOR_SIZE},
            ))
            report.status = ANEStatus.PARTIAL

        report.issues.extend(issues)
        report.notes.append("Linear should be converted to Conv2d for optimal ANE performance")

    def _check_conv_layer(self, module: Union[nn.Conv1d, nn.Conv2d], report: ANELayerReport):
        """Check Conv layer for ANE compatibility."""
        in_channels = module.in_channels
        out_channels = module.out_channels
        kernel_size = module.kernel_size
        groups = module.groups

        report.status = ANEStatus.OPTIMIZED
        report.notes.append("Conv layer is already ANE-optimized format")

        issues = []

        # Check channel alignment
        if in_channels % 4 != 0:
            issues.append(ANEIssue(
                level=ANEIssueLevel.INFO,
                layer_name=report.name,
                layer_type=report.type,
                message=f"Input channels ({in_channels}) not divisible by 4",
                suggestion="Padding to multiple of 4 may improve performance",
                details={"in_channels": in_channels},
            ))

        if out_channels % 4 != 0:
            issues.append(ANEIssue(
                level=ANEIssueLevel.INFO,
                layer_name=report.name,
                layer_type=report.type,
                message=f"Output channels ({out_channels}) not divisible by 4",
                suggestion="Padding to multiple of 4 may improve performance",
                details={"out_channels": out_channels},
            ))

        # Check for depthwise conv
        if groups == in_channels and groups == out_channels:
            report.notes.append("Depthwise convolution - efficient on ANE")

        report.issues.extend(issues)

    def _check_layernorm(self, module: nn.LayerNorm, report: ANELayerReport):
        """Check LayerNorm for ANE compatibility."""
        normalized_shape = module.normalized_shape

        report.status = ANEStatus.COMPATIBLE

        # ANE implements LayerNorm efficiently
        report.notes.append("LayerNorm is ANE compatible")

        if isinstance(normalized_shape, (list, tuple)):
            dim = normalized_shape[-1] if len(normalized_shape) > 0 else normalized_shape
        else:
            dim = normalized_shape

        if dim % 4 != 0:
            report.issues.append(ANEIssue(
                level=ANEIssueLevel.INFO,
                layer_name=report.name,
                layer_type="LayerNorm",
                message=f"Normalized dimension ({dim}) not divisible by 4",
                suggestion="May have slightly reduced efficiency",
                details={"normalized_shape": normalized_shape},
            ))

    def _check_embedding(self, module: nn.Embedding, report: ANELayerReport):
        """Check Embedding layer for ANE compatibility."""
        num_embeddings = module.num_embeddings
        embedding_dim = module.embedding_dim

        report.input_shape = (num_embeddings,)
        report.output_shape = (embedding_dim,)

        # Large embeddings may fall back to CPU
        total_size = num_embeddings * embedding_dim

        if total_size > ANE_MAX_TENSOR_SIZE:
            report.status = ANEStatus.PARTIAL
            report.issues.append(ANEIssue(
                level=ANEIssueLevel.WARNING,
                layer_name=report.name,
                layer_type="Embedding",
                message=f"Large embedding table ({total_size:,} elements) may fall back to CPU",
                suggestion="Consider using smaller vocabulary or embedding dimension",
                details={"num_embeddings": num_embeddings, "embedding_dim": embedding_dim},
            ))
        else:
            report.status = ANEStatus.COMPATIBLE
            report.notes.append("Embedding size is within ANE limits")

        if embedding_dim % 4 != 0:
            report.issues.append(ANEIssue(
                level=ANEIssueLevel.INFO,
                layer_name=report.name,
                layer_type="Embedding",
                message=f"Embedding dimension ({embedding_dim}) not divisible by 4",
                suggestion="Padding may improve ANE efficiency",
                details={"embedding_dim": embedding_dim},
            ))

    def _check_softmax(self, module: nn.Softmax, report: ANELayerReport):
        """Check Softmax for ANE compatibility."""
        dim = module.dim

        report.status = ANEStatus.COMPATIBLE
        report.notes.append("Softmax is ANE compatible")

        # Softmax over very large dimensions may be slow
        report.issues.append(ANEIssue(
            level=ANEIssueLevel.INFO,
            layer_name=report.name,
            layer_type="Softmax",
            message=f"Softmax over dim={dim}",
            suggestion="Ensure the dimension is not excessively large for best performance",
            details={"dim": dim},
        ))

    def _check_recurrent_layer(self, module: nn.Module, report: ANELayerReport):
        """Check RNN/LSTM/GRU for ANE compatibility."""
        report.status = ANEStatus.INCOMPATIBLE

        report.issues.append(ANEIssue(
            level=ANEIssueLevel.ERROR,
            layer_name=report.name,
            layer_type=report.type,
            message="Recurrent layers (LSTM/GRU/RNN) are not supported on ANE",
            suggestion="Consider using Transformer-based architecture or running on GPU",
            details={"layer_type": report.type},
        ))

        report.notes.append("Recurrent layers will run on CPU/GPU, not ANE")

    def _check_attention(self, module: nn.MultiheadAttention, report: ANELayerReport):
        """Check MultiheadAttention for ANE compatibility."""
        embed_dim = module.embed_dim
        num_heads = module.num_heads
        head_dim = embed_dim // num_heads

        report.status = ANEStatus.NEEDS_CONVERSION

        issues = []

        # Check head dimension alignment
        if head_dim % 8 != 0:
            issues.append(ANEIssue(
                level=ANEIssueLevel.WARNING,
                layer_name=report.name,
                layer_type="MultiheadAttention",
                message=f"Head dimension ({head_dim}) not divisible by 8",
                suggestion="Use head_dim divisible by 8 for optimal ANE performance",
                details={"embed_dim": embed_dim, "num_heads": num_heads, "head_dim": head_dim},
            ))

        if embed_dim % 4 != 0:
            issues.append(ANEIssue(
                level=ANEIssueLevel.WARNING,
                layer_name=report.name,
                layer_type="MultiheadAttention",
                message=f"Embed dimension ({embed_dim}) not divisible by 4",
                suggestion="Padding may improve performance",
                details={"embed_dim": embed_dim},
            ))

        report.issues.extend(issues)
        report.notes.append("Attention should use ANE-optimized implementation (Conv2d-based)")

    def _generate_summary(self, report: ANEReport) -> Dict[str, Any]:
        """Generate summary statistics for the report."""
        # Count issues by level
        issue_counts = {level: 0 for level in ANEIssueLevel}
        for issue in report.issues:
            issue_counts[issue.level] += 1

        # Calculate compatibility score
        score = report.get_compatibility_score()

        # Determine overall recommendation
        if score >= 90:
            recommendation = "Excellent ANE compatibility. Model should run efficiently on ANE."
        elif score >= 70:
            recommendation = "Good ANE compatibility. Minor optimizations recommended."
        elif score >= 50:
            recommendation = "Moderate ANE compatibility. Some layers may fall back to CPU/GPU."
        else:
            recommendation = "Poor ANE compatibility. Consider architectural changes."

        return {
            "compatibility_score": score,
            "recommendation": recommendation,
            "issue_counts": {
                "info": issue_counts[ANEIssueLevel.INFO],
                "warning": issue_counts[ANEIssueLevel.WARNING],
                "error": issue_counts[ANEIssueLevel.ERROR],
            },
            "layer_status_counts": {
                "optimized": report.optimized_layers,
                "compatible": report.compatible_layers,
                "needs_conversion": report.needs_conversion_layers,
                "partial": report.partial_layers,
                "incompatible": report.incompatible_layers,
            },
        }

    def check_coreml_model(self, model_path: str) -> ANEReport:
        """Check a CoreML model for ANE compatibility.

        Args:
            model_path: Path to .mlpackage or .mlmodel file

        Returns:
            ANEReport with compatibility analysis
        """
        try:
            import coremltools as ct
        except ImportError:
            raise ImportError("coremltools required. Install with: pip install coremltools")

        model_name = os.path.basename(model_path)
        report = ANEReport(
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
        )

        if self.verbose:
            print(f"Checking CoreML model: {model_path}")

        # Load model
        model = ct.models.MLModel(model_path)
        spec = model.get_spec()

        # Check compute units
        if hasattr(spec, 'computeUnits'):
            compute_units = spec.computeUnits
            report.summary["compute_units"] = str(compute_units)

        # Analyze program (ML Program format)
        if spec.WhichOneof('Type') == 'mlProgram':
            self._check_mlprogram(model, report)
        else:
            # Neural network format
            self._check_neural_network(spec, report)

        # Generate summary
        report.summary.update(self._generate_summary(report))

        return report

    def _check_mlprogram(self, model, report: ANEReport):
        """Check ML Program format CoreML model."""
        try:
            from coremltools.converters.mil.frontend.milproto.load import load as mil_load

            spec = model.get_spec()
            prog = mil_load(
                spec,
                specification_version=spec.specificationVersion,
                file_weights_dir=model.weights_dir,
            )

            # Analyze operations
            for func_name, func in prog.functions.items():
                for op in func.operations:
                    layer_report = self._check_mil_op(op, func_name)
                    report.layers.append(layer_report)
                    report.total_layers += 1

                    # Update counters
                    if layer_report.status == ANEStatus.COMPATIBLE:
                        report.compatible_layers += 1
                    elif layer_report.status == ANEStatus.OPTIMIZED:
                        report.optimized_layers += 1
                    elif layer_report.status == ANEStatus.NEEDS_CONVERSION:
                        report.needs_conversion_layers += 1
                    elif layer_report.status == ANEStatus.PARTIAL:
                        report.partial_layers += 1
                    elif layer_report.status == ANEStatus.INCOMPATIBLE:
                        report.incompatible_layers += 1

                    report.issues.extend(layer_report.issues)

        except Exception as e:
            report.issues.append(ANEIssue(
                level=ANEIssueLevel.WARNING,
                layer_name="model",
                layer_type="mlprogram",
                message=f"Could not analyze ML Program: {e}",
                suggestion="Model may still be ANE compatible",
            ))

    def _check_mil_op(self, op, func_name: str) -> ANELayerReport:
        """Check a MIL operation for ANE compatibility."""
        op_type = op.op_type
        op_name = f"{func_name}/{op.name}"

        layer_report = ANELayerReport(
            name=op_name,
            type=op_type,
            status=ANEStatus.UNKNOWN,
        )

        # Check by operation type
        op_lower = op_type.lower()

        if op_lower in ["const", "constexpr_lut_to_dense"]:
            layer_report.status = ANEStatus.COMPATIBLE
            layer_report.notes.append("Constant operation")

        elif op_lower in ["conv", "conv_transpose"]:
            layer_report.status = ANEStatus.OPTIMIZED
            layer_report.notes.append("Convolution is ANE-optimized")

        elif op_lower == "linear":
            layer_report.status = ANEStatus.NEEDS_CONVERSION
            layer_report.notes.append("Linear should be Conv2d for ANE")
            layer_report.issues.append(ANEIssue(
                level=ANEIssueLevel.INFO,
                layer_name=op_name,
                layer_type=op_type,
                message="Linear operation found",
                suggestion="Convert to Conv2d for better ANE efficiency",
            ))

        elif op_lower in ["matmul", "bmm"]:
            layer_report.status = ANEStatus.COMPATIBLE
            layer_report.notes.append("Matrix multiplication is ANE compatible")

        elif op_lower in ["relu", "gelu", "silu", "sigmoid", "tanh", "softmax"]:
            layer_report.status = ANEStatus.COMPATIBLE
            layer_report.notes.append("Activation is ANE compatible")

        elif op_lower in ["add", "sub", "mul", "div", "real_div"]:
            layer_report.status = ANEStatus.COMPATIBLE
            layer_report.notes.append("Element-wise operation is ANE compatible")

        elif op_lower in ["layer_norm", "instance_norm", "batch_norm"]:
            layer_report.status = ANEStatus.COMPATIBLE
            layer_report.notes.append("Normalization is ANE compatible")

        elif op_lower in ["reshape", "transpose", "expand_dims", "squeeze"]:
            layer_report.status = ANEStatus.COMPATIBLE
            layer_report.notes.append("Shape operation is ANE compatible")

        elif op_lower in ["gather", "scatter"]:
            layer_report.status = ANEStatus.PARTIAL
            layer_report.notes.append("Gather/Scatter may fall back to CPU")
            layer_report.issues.append(ANEIssue(
                level=ANEIssueLevel.WARNING,
                layer_name=op_name,
                layer_type=op_type,
                message="Gather/Scatter operations may fall back to CPU",
                suggestion="Consider alternative implementations if performance critical",
            ))

        elif op_lower == "einsum":
            layer_report.status = ANEStatus.PARTIAL
            layer_report.notes.append("Einsum may fall back to CPU")
            layer_report.issues.append(ANEIssue(
                level=ANEIssueLevel.WARNING,
                layer_name=op_name,
                layer_type=op_type,
                message="Einsum operations may not be fully ANE optimized",
                suggestion="Consider using explicit matmul operations",
            ))

        elif op_lower in ["lstm", "gru", "rnn"]:
            layer_report.status = ANEStatus.INCOMPATIBLE
            layer_report.notes.append("RNN operations not supported on ANE")
            layer_report.issues.append(ANEIssue(
                level=ANEIssueLevel.ERROR,
                layer_name=op_name,
                layer_type=op_type,
                message="Recurrent operations are not supported on ANE",
                suggestion="Use Transformer architecture instead",
            ))

        else:
            layer_report.status = ANEStatus.UNKNOWN
            layer_report.notes.append(f"Unknown operation type: {op_type}")

        return layer_report

    def _check_neural_network(self, spec, report: ANEReport):
        """Check Neural Network format CoreML model."""
        if not hasattr(spec, 'neuralNetwork'):
            report.issues.append(ANEIssue(
                level=ANEIssueLevel.WARNING,
                layer_name="model",
                layer_type="unknown",
                message="Not a neural network model",
                suggestion="Model type may not be fully analyzable",
            ))
            return

        nn = spec.neuralNetwork

        for layer in nn.layers:
            layer_type = layer.WhichOneof('layer')
            layer_name = layer.name

            layer_report = ANELayerReport(
                name=layer_name,
                type=layer_type or "unknown",
                status=ANEStatus.UNKNOWN,
            )

            # Basic layer type checking
            if layer_type in ["convolution", "innerProduct"]:
                if layer_type == "convolution":
                    layer_report.status = ANEStatus.OPTIMIZED
                else:
                    layer_report.status = ANEStatus.NEEDS_CONVERSION

            elif layer_type in ["activation", "batchnorm", "pooling"]:
                layer_report.status = ANEStatus.COMPATIBLE

            elif layer_type in ["uniDirectionalLSTM", "biDirectionalLSTM"]:
                layer_report.status = ANEStatus.INCOMPATIBLE
                layer_report.issues.append(ANEIssue(
                    level=ANEIssueLevel.ERROR,
                    layer_name=layer_name,
                    layer_type=layer_type,
                    message="LSTM not supported on ANE",
                    suggestion="Use Transformer architecture",
                ))

            report.layers.append(layer_report)
            report.total_layers += 1

            # Update counters
            if layer_report.status == ANEStatus.COMPATIBLE:
                report.compatible_layers += 1
            elif layer_report.status == ANEStatus.OPTIMIZED:
                report.optimized_layers += 1
            elif layer_report.status == ANEStatus.NEEDS_CONVERSION:
                report.needs_conversion_layers += 1
            elif layer_report.status == ANEStatus.PARTIAL:
                report.partial_layers += 1
            elif layer_report.status == ANEStatus.INCOMPATIBLE:
                report.incompatible_layers += 1

            report.issues.extend(layer_report.issues)

    def save_report(
        self,
        report: ANEReport,
        output_path: str,
        format: str = "txt",
    ):
        """Save ANE compatibility report to file.

        Args:
            report: ANEReport to save
            output_path: Output file path
            format: Output format ('txt' or 'json')
        """
        if format == "json":
            self._save_json_report(report, output_path)
        else:
            self._save_txt_report(report, output_path)

        if self.verbose:
            print(f"Report saved to: {output_path}")

    def _save_txt_report(self, report: ANEReport, output_path: str):
        """Save report as formatted text file."""
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("APPLE NEURAL ENGINE (ANE) COMPATIBILITY REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Model: {report.model_name}")
        lines.append(f"Generated: {report.timestamp}")
        lines.append("")

        # Summary
        lines.append("-" * 80)
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append("")

        score = report.get_compatibility_score()
        lines.append(f"ANE Compatibility Score: {score:.1f}/100")
        lines.append("")

        if "recommendation" in report.summary:
            lines.append(f"Recommendation: {report.summary['recommendation']}")
            lines.append("")

        lines.append("Layer Statistics:")
        lines.append(f"  Total layers:         {report.total_layers}")
        lines.append(f"  Optimized (ANE):      {report.optimized_layers}")
        lines.append(f"  Compatible:           {report.compatible_layers}")
        lines.append(f"  Needs conversion:     {report.needs_conversion_layers}")
        lines.append(f"  Partial support:      {report.partial_layers}")
        lines.append(f"  Incompatible:         {report.incompatible_layers}")
        lines.append("")

        # Issue summary
        if "issue_counts" in report.summary:
            counts = report.summary["issue_counts"]
            lines.append("Issue Summary:")
            lines.append(f"  Errors:   {counts['error']}")
            lines.append(f"  Warnings: {counts['warning']}")
            lines.append(f"  Info:     {counts['info']}")
            lines.append("")

        # Detailed issues
        if report.issues:
            lines.append("-" * 80)
            lines.append("ISSUES")
            lines.append("-" * 80)
            lines.append("")

            # Group by level
            errors = [i for i in report.issues if i.level == ANEIssueLevel.ERROR]
            warnings = [i for i in report.issues if i.level == ANEIssueLevel.WARNING]
            infos = [i for i in report.issues if i.level == ANEIssueLevel.INFO]

            if errors:
                lines.append("ERRORS (Will not run on ANE):")
                for issue in errors:
                    lines.append(f"  [{issue.layer_name}] {issue.message}")
                    if issue.suggestion:
                        lines.append(f"    -> Suggestion: {issue.suggestion}")
                lines.append("")

            if warnings:
                lines.append("WARNINGS (May impact performance):")
                for issue in warnings:
                    lines.append(f"  [{issue.layer_name}] {issue.message}")
                    if issue.suggestion:
                        lines.append(f"    -> Suggestion: {issue.suggestion}")
                lines.append("")

            if infos:
                lines.append("INFO (Informational):")
                for issue in infos:
                    lines.append(f"  [{issue.layer_name}] {issue.message}")
                lines.append("")

        # Layer details
        lines.append("-" * 80)
        lines.append("LAYER DETAILS")
        lines.append("-" * 80)
        lines.append("")

        # Group by status
        status_groups = {}
        for layer in report.layers:
            status_groups.setdefault(layer.status, []).append(layer)

        status_order = [
            ANEStatus.INCOMPATIBLE,
            ANEStatus.PARTIAL,
            ANEStatus.NEEDS_CONVERSION,
            ANEStatus.COMPATIBLE,
            ANEStatus.OPTIMIZED,
            ANEStatus.UNKNOWN,
        ]

        for status in status_order:
            if status not in status_groups:
                continue

            layers = status_groups[status]
            lines.append(f"[{status.value.upper()}] ({len(layers)} layers)")

            for layer in layers:
                line = f"  {layer.name}: {layer.type}"
                if layer.param_count > 0:
                    line += f" ({layer.param_count:,} params)"
                lines.append(line)

                for note in layer.notes:
                    lines.append(f"    - {note}")

            lines.append("")

        # Footer
        lines.append("-" * 80)
        lines.append("END OF REPORT")
        lines.append("-" * 80)

        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

    def _save_json_report(self, report: ANEReport, output_path: str):
        """Save report as JSON file."""
        data = {
            "model_name": report.model_name,
            "timestamp": report.timestamp,
            "compatibility_score": report.get_compatibility_score(),
            "summary": report.summary,
            "statistics": {
                "total_layers": report.total_layers,
                "optimized_layers": report.optimized_layers,
                "compatible_layers": report.compatible_layers,
                "needs_conversion_layers": report.needs_conversion_layers,
                "partial_layers": report.partial_layers,
                "incompatible_layers": report.incompatible_layers,
            },
            "issues": [
                {
                    "level": issue.level.value,
                    "layer_name": issue.layer_name,
                    "layer_type": issue.layer_type,
                    "message": issue.message,
                    "suggestion": issue.suggestion,
                    "details": issue.details,
                }
                for issue in report.issues
            ],
            "layers": [
                {
                    "name": layer.name,
                    "type": layer.type,
                    "status": layer.status.value,
                    "param_count": layer.param_count,
                    "notes": layer.notes,
                }
                for layer in report.layers
            ],
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


def check_ane_compatibility(
    model: Union[nn.Module, str],
    output_path: str = None,
    verbose: bool = True,
) -> ANEReport:
    """Convenience function to check ANE compatibility.

    Args:
        model: PyTorch model or path to CoreML model
        output_path: Optional path to save report (.txt or .json)
        verbose: Print progress information

    Returns:
        ANEReport with compatibility analysis
    """
    checker = ANEChecker(verbose=verbose)

    if isinstance(model, str):
        # CoreML model path
        report = checker.check_coreml_model(model)
    else:
        # PyTorch model
        report = checker.check_pytorch_model(model)

    if output_path:
        format = "json" if output_path.endswith(".json") else "txt"
        checker.save_report(report, output_path, format=format)

    return report
