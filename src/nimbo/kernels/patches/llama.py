"""LLaMA model patcher for Nimbo Triton kernels.

Supports:
- LLaMA 2 (7B, 13B, 70B)
- LLaMA 3 (8B, 70B)
- Code Llama
- Mistral (similar architecture)
- Phi (similar architecture)

LLaMA Architecture:
- GQA (Grouped Query Attention) in LLaMA 2 70B and LLaMA 3
- MHA (Multi-Head Attention) in LLaMA 2 7B/13B
- RMSNorm with eps=1e-5
- SwiGLU activation
- RoPE
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from .base import ModelPatcher, register_patcher, PatchStats
from ..triton.rms_norm import NimboRMSNorm
from ..triton.swiglu import swiglu
from ..triton.rope import NimboRoPE

logger = logging.getLogger(__name__)


class PatchedLlamaMLP(nn.Module):
    """Patched MLP layer using Triton SwiGLU kernel."""

    def __init__(self, original_mlp: nn.Module):
        super().__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = swiglu(gate, up)
        return self.down_proj(hidden)


@register_patcher
class LlamaPatcher(ModelPatcher):
    """Patcher for LLaMA-family models."""

    SUPPORTED_MODELS = ["llama", "mistral", "phi", "phi3", "qwen2"]

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.config = model.config

    def patch_rms_norm(self) -> int:
        """Patch all RMSNorm layers."""
        count = 0

        for name, module in self.model.named_modules():
            module_type = type(module).__name__

            if "RMSNorm" in module_type or module_type == "LlamaRMSNorm":
                eps = getattr(module, 'variance_epsilon',
                              getattr(module, 'eps', 1e-5))

                nimbo_norm = NimboRMSNorm(
                    hidden_size=module.weight.shape[0],
                    eps=eps,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nimbo_norm.weight.data = module.weight.data.clone()

                self._store_original(name, module)
                self._set_module(name, nimbo_norm)

                count += 1
                logger.debug(f"Patched RMSNorm: {name}")

        return count

    def patch_swiglu(self) -> int:
        """Patch all MLP layers."""
        count = 0

        for name, module in self.model.named_modules():
            module_type = type(module).__name__

            if "MLP" in module_type and hasattr(module, 'gate_proj'):
                try:
                    patched_mlp = PatchedLlamaMLP(module)
                    self._store_original(name, module)
                    self._set_module(name, patched_mlp)
                    count += 1
                    logger.debug(f"Patched MLP: {name}")
                except Exception as e:
                    logger.warning(f"Failed to patch MLP {name}: {e}")

        return count

    def patch_rope(self) -> int:
        """Patch RoPE embeddings."""
        count = 0

        for name, module in self.model.named_modules():
            module_type = type(module).__name__

            if "RotaryEmbedding" in module_type:
                rope_config = getattr(self.config, 'rope_scaling', {}) or {}

                patched_rope = NimboRoPE(
                    head_dim=self.config.hidden_size // self.config.num_attention_heads,
                    max_seq_len=getattr(self.config, 'max_position_embeddings', 4096),
                    base=getattr(self.config, 'rope_theta', 10000.0),
                    scaling_factor=rope_config.get('factor', 1.0),
                    rope_type=rope_config.get('rope_type', 'default'),
                    device=next(self.model.parameters()).device,
                )

                self._store_original(name, module)
                self._set_module(name, patched_rope)

                count += 1
                logger.debug(f"Patched RoPE: {name}")

        return count

    def patch_attention(self) -> int:
        """Patch attention mechanism."""
        # Rely on Flash Attention 2 for now
        logger.info("Attention patching relies on Flash Attention 2. "
                    "Use use_flash_attention=True in Nimbo.")
        return 0


def patch_llama(
    model: nn.Module,
    rms_norm: bool = True,
    swiglu: bool = True,
    rope: bool = True,
    attention: bool = False,
) -> PatchStats:
    """Apply Nimbo patches to LLaMA-family models.

    Args:
        model: LLaMA/Mistral/Phi model to patch
        rms_norm: Patch RMSNorm layers
        swiglu: Patch SwiGLU MLP layers
        rope: Patch RoPE embeddings
        attention: Patch attention mechanism

    Returns:
        PatchStats with counts of patched layers
    """
    patcher = LlamaPatcher(model)
    return patcher.patch_all(
        rms_norm=rms_norm,
        swiglu=swiglu,
        rope=rope,
        attention=attention,
    )
