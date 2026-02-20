"""EXAONE model patcher for Nimbo Triton kernels.

Supports:
- EXAONE 3.5 (2.4B, 7.8B, 32B)
- EXAONE 4.0 (1.2B, 32B)

EXAONE Architecture:
- GQA (Grouped Query Attention): 32:8 ratio (4:1)
- RMSNorm with eps=1e-5
- SwiGLU activation (SiLU gated)
- RoPE with Llama3-style scaling (base=1M, factor=16.0 for 4.0)
"""

import logging
import sys
from typing import Optional, Callable
import functools

import torch
import torch.nn as nn

from .base import ModelPatcher, register_patcher, PatchStats
from ..triton.rms_norm import NimboRMSNorm
from ..triton.swiglu import NimboSwiGLU, swiglu
from ..triton.rope import NimboRoPE, apply_rotary_pos_emb
from ..triton.rope_fused import apply_rotary_pos_emb_triton

logger = logging.getLogger(__name__)

# Store original functions for unpatching
_ORIGINAL_ROPE_FUNCS = {}


class PatchedExaoneMLP(nn.Module):
    """Patched MLP layer using Triton SwiGLU kernel."""

    def __init__(self, original_mlp: nn.Module):
        super().__init__()
        # Copy linear layers from original
        self.gate_proj = original_mlp.c_fc_0 if hasattr(original_mlp, 'c_fc_0') else original_mlp.gate_proj
        self.up_proj = original_mlp.c_fc_1 if hasattr(original_mlp, 'c_fc_1') else original_mlp.up_proj
        self.down_proj = original_mlp.c_proj if hasattr(original_mlp, 'c_proj') else original_mlp.down_proj

        # Store activation function name for reference
        self.act_fn_name = "silu"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply gate and up projections
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # Fused SwiGLU using Triton kernel
        hidden = swiglu(gate, up)

        # Down projection
        return self.down_proj(hidden)


class PatchedExaoneRoPE(nn.Module):
    """Patched RoPE using Triton kernel with EXAONE-specific settings."""

    def __init__(
        self,
        config,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        # Extract EXAONE RoPE config
        self.head_dim = config.hidden_size // config.num_attention_heads

        # EXAONE uses Llama3-style scaling
        rope_config = getattr(config, 'rope_scaling', {}) or {}

        self.nimbo_rope = NimboRoPE(
            head_dim=self.head_dim,
            max_seq_len=getattr(config, 'max_position_embeddings', 32768),
            base=getattr(config, 'rope_theta', 1000000.0),
            scaling_factor=rope_config.get('factor', 8.0),
            rope_type=rope_config.get('rope_type', 'llama3'),
            original_max_position=rope_config.get('original_max_position_embeddings', 8192),
            high_freq_factor=rope_config.get('high_freq_factor', 4.0),
            low_freq_factor=rope_config.get('low_freq_factor', 1.0),
            device=device,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ):
        return self.nimbo_rope(q, k, position_ids)


@register_patcher
class ExaonePatcher(ModelPatcher):
    """Patcher for EXAONE models."""

    SUPPORTED_MODELS = ["exaone", "exaone4"]

    def __init__(self, model: nn.Module):
        super().__init__(model)

        # Detect EXAONE version
        self.is_exaone4 = "exaone4" in model.__class__.__name__.lower()

        # Get config
        self.config = model.config

        # Layer name patterns differ between versions
        if self.is_exaone4:
            self.layer_pattern = "model.layers"
            self.norm_pattern = "model.norm"
        else:
            self.layer_pattern = "transformer.h"
            self.norm_pattern = "transformer.ln_f"

    def patch_rms_norm(self) -> int:
        """Patch all RMSNorm layers in EXAONE model."""
        count = 0

        for name, module in self.model.named_modules():
            # Check for EXAONE RMSNorm variants
            module_type = type(module).__name__
            if module_type in ["ExaoneRMSNorm", "RMSNorm", "Exaone4RMSNorm"]:
                # Get epsilon
                eps = getattr(module, 'eps', getattr(module, 'variance_epsilon', 1e-5))

                # Create Nimbo RMSNorm
                nimbo_norm = NimboRMSNorm(
                    hidden_size=module.weight.shape[0],
                    eps=eps,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )

                # Copy weight
                nimbo_norm.weight.data = module.weight.data.clone()

                # Store original and replace
                self._store_original(name, module)
                self._set_module(name, nimbo_norm)

                count += 1
                logger.debug(f"Patched RMSNorm: {name}")

        return count

    def patch_swiglu(self) -> int:
        """Patch all MLP layers to use Triton SwiGLU."""
        count = 0

        for name, module in self.model.named_modules():
            module_type = type(module).__name__

            # EXAONE 3.5 uses ExaoneGatedMLP, EXAONE 4.0 uses Exaone4MLP
            if module_type in ["ExaoneGatedMLP", "ExaoneMLP", "Exaone4MLP"]:
                try:
                    patched_mlp = PatchedExaoneMLP(module)

                    self._store_original(name, module)
                    self._set_module(name, patched_mlp)

                    count += 1
                    logger.debug(f"Patched MLP: {name}")
                except Exception as e:
                    logger.warning(f"Failed to patch MLP {name}: {e}")

        return count

    def patch_rope(self) -> int:
        """Patch RoPE to use Triton kernel.

        For EXAONE 4.0, we monkey-patch the module-level apply_rotary_pos_emb function.
        For EXAONE 3.5, we patch the rotary_emb module in each attention layer.
        """
        count = 0

        if self.is_exaone4:
            # EXAONE 4.0: Monkey-patch the module-level function
            count = self._patch_rope_exaone4()
        else:
            # EXAONE 3.5: Patch rotary_emb modules
            count = self._patch_rope_exaone35()

        return count

    def _patch_rope_exaone4(self) -> int:
        """Patch RoPE for EXAONE 4.0 by replacing the module-level function."""
        try:
            import transformers.models.exaone4.modeling_exaone4 as exaone4_module

            # Store original function
            if 'exaone4' not in _ORIGINAL_ROPE_FUNCS:
                _ORIGINAL_ROPE_FUNCS['exaone4'] = exaone4_module.apply_rotary_pos_emb

            # Replace with Triton version
            exaone4_module.apply_rotary_pos_emb = apply_rotary_pos_emb_triton

            # Count attention layers (each will use the patched function)
            count = 0
            for name, module in self.model.named_modules():
                if "Exaone4Attention" in type(module).__name__:
                    count += 1

            logger.info(f"Patched EXAONE 4.0 apply_rotary_pos_emb function ({count} attention layers)")
            return count

        except ImportError as e:
            logger.warning(f"Could not patch EXAONE 4.0 RoPE: {e}")
            return 0

    def _patch_rope_exaone35(self) -> int:
        """Patch RoPE for EXAONE 3.5 by replacing rotary_emb modules."""
        count = 0

        for name, module in self.model.named_modules():
            module_type = type(module).__name__

            if "Attention" in module_type and "Exaone" in module_type:
                if hasattr(module, 'rotary_emb'):
                    original_rotary = module.rotary_emb

                    patched_rope = PatchedExaoneRoPE(
                        self.config,
                        device=next(module.parameters()).device,
                    )

                    self._store_original(f"{name}.rotary_emb", original_rotary)
                    module.rotary_emb = patched_rope

                    count += 1
                    logger.debug(f"Patched RoPE in: {name}")

        return count

    def patch_attention(self) -> int:
        """Patch attention mechanism (advanced, uses Flash Attention)."""
        # For now, we rely on HuggingFace's Flash Attention 2 support
        # Custom GQA kernel would go here
        count = 0

        # Check if model is using Flash Attention
        if getattr(self.config, '_attn_implementation', None) == 'flash_attention_2':
            logger.info("Model already using Flash Attention 2")
            return count

        # TODO: Implement custom GQA kernel for EXAONE
        # This would include:
        # - Fused QKV projection
        # - GQA with 4:1 ratio optimization
        # - KV cache management

        logger.info("Custom attention patching not yet implemented. "
                    "Use use_flash_attention=True in Nimbo for best performance.")

        return count


def patch_exaone(
    model: nn.Module,
    rms_norm: bool = True,
    swiglu: bool = True,
    rope: bool = True,
    attention: bool = False,
) -> PatchStats:
    """Apply Nimbo Triton kernel patches to EXAONE model.

    Args:
        model: EXAONE model to patch
        rms_norm: Patch RMSNorm layers (7-8x speedup)
        swiglu: Patch SwiGLU MLP layers (3-5x speedup)
        rope: Patch RoPE embeddings (1.9-2.3x speedup)
        attention: Patch attention mechanism (requires Flash Attention)

    Returns:
        PatchStats with counts of patched layers

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
        >>> from nimbo.kernels import patch_exaone
        >>> stats = patch_exaone(model)
        >>> print(stats)
    """
    patcher = ExaonePatcher(model)
    return patcher.patch_all(
        rms_norm=rms_norm,
        swiglu=swiglu,
        rope=rope,
        attention=attention,
    )
