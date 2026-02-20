"""Nimbo Custom Triton Kernels for LLM Training Acceleration.

This module provides optimized Triton kernels for various LLM operations:
- RMSNorm: Fused forward and backward pass (7-8x speedup)
- SwiGLU: Fused gated MLP activation (3-5x speedup)
- RoPE: Fused rotary position embedding for Q and K (1.9-2.3x speedup)

Supported model architectures:
- EXAONE (3.5, 4.0) - Full support with EXAONE-specific optimizations
- LLaMA (2, 3) - Full support
- Mistral - Full support
- Phi (2, 3) - Full support
- Qwen (2) - Full support

Example:
    >>> from nimbo.kernels import patch_exaone
    >>> from transformers import AutoModelForCausalLM
    >>>
    >>> model = AutoModelForCausalLM.from_pretrained("LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
    >>> stats = patch_exaone(model)
    >>> print(stats)
    Nimbo patches applied to ExaonePatcher:
      - RMSNorm: 65
      - SwiGLU: 32
      - RoPE: 32
      - Total: 129
"""

import logging

logger = logging.getLogger(__name__)

# Check if Triton is available
_TRITON_AVAILABLE = False
try:
    import triton
    _TRITON_AVAILABLE = True
except ImportError:
    logger.warning(
        "Triton not available. Nimbo kernels will not be loaded. "
        "Install with: pip install triton"
    )


def is_triton_available() -> bool:
    """Check if Triton is available."""
    return _TRITON_AVAILABLE


if _TRITON_AVAILABLE:
    from .triton import (
        triton_rms_norm,
        triton_rms_norm_backward,
        triton_swiglu_forward,
        triton_swiglu_backward,
        triton_rope_forward,
        triton_rope_backward,
        NimboRMSNorm,
        NimboSwiGLU,
        NimboRoPE,
    )

    from .patches import (
        patch_exaone,
        patch_llama,
        unpatch_model,
        get_supported_models,
    )

    from .patches.base import patch_model

    __all__ = [
        # Availability check
        "is_triton_available",
        # Triton kernels
        "triton_rms_norm",
        "triton_rms_norm_backward",
        "triton_swiglu_forward",
        "triton_swiglu_backward",
        "triton_rope_forward",
        "triton_rope_backward",
        # Module replacements
        "NimboRMSNorm",
        "NimboSwiGLU",
        "NimboRoPE",
        # Model patchers
        "patch_model",
        "patch_exaone",
        "patch_llama",
        "unpatch_model",
        "get_supported_models",
    ]
else:
    __all__ = ["is_triton_available"]

    def _not_available(*args, **kwargs):
        raise ImportError(
            "Triton is required for Nimbo kernels. "
            "Install with: pip install triton"
        )

    triton_rms_norm = _not_available
    triton_rms_norm_backward = _not_available
    triton_swiglu_forward = _not_available
    triton_swiglu_backward = _not_available
    triton_rope_forward = _not_available
    triton_rope_backward = _not_available
    patch_exaone = _not_available
    patch_llama = _not_available
    patch_model = _not_available
    unpatch_model = _not_available
    get_supported_models = lambda: []
