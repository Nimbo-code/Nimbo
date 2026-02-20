"""Triton kernel implementations for Nimbo."""

from .rms_norm import (
    triton_rms_norm,
    triton_rms_norm_backward,
    NimboRMSNorm,
)

from .swiglu import (
    triton_swiglu_forward,
    triton_swiglu_backward,
    NimboSwiGLU,
    swiglu,
)

from .rope import (
    triton_rope_forward,
    triton_rope_backward,
    NimboRoPE,
    apply_rotary_pos_emb,
)

from .rope_fused import (
    triton_apply_rotary_pos_emb,
    apply_rotary_pos_emb_triton,
)

__all__ = [
    # RMSNorm
    "triton_rms_norm",
    "triton_rms_norm_backward",
    "NimboRMSNorm",
    # SwiGLU
    "triton_swiglu_forward",
    "triton_swiglu_backward",
    "NimboSwiGLU",
    "swiglu",
    # RoPE
    "triton_rope_forward",
    "triton_rope_backward",
    "NimboRoPE",
    "apply_rotary_pos_emb",
    # Fused RoPE (for HuggingFace models)
    "triton_apply_rotary_pos_emb",
    "apply_rotary_pos_emb_triton",
]
