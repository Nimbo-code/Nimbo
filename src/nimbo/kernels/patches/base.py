"""Base model patcher for Nimbo Triton kernels."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type, Any
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Registry of supported models
_PATCHERS: Dict[str, Type["ModelPatcher"]] = {}

# Store original modules for unpatching
_ORIGINAL_MODULES: Dict[int, Dict[str, nn.Module]] = {}


@dataclass
class PatchStats:
    """Statistics about applied patches."""

    model_type: str
    rms_norm_patched: int = 0
    swiglu_patched: int = 0
    rope_patched: int = 0
    attention_patched: int = 0
    total_patched: int = 0

    def __str__(self) -> str:
        return (
            f"Nimbo patches applied to {self.model_type}:\n"
            f"  - RMSNorm: {self.rms_norm_patched}\n"
            f"  - SwiGLU: {self.swiglu_patched}\n"
            f"  - RoPE: {self.rope_patched}\n"
            f"  - Attention: {self.attention_patched}\n"
            f"  - Total: {self.total_patched}"
        )


class ModelPatcher(ABC):
    """Base class for model-specific patchers."""

    # Model identifiers that this patcher handles
    SUPPORTED_MODELS: List[str] = []

    def __init__(self, model: nn.Module):
        self.model = model
        self.model_id = id(model)
        self.stats = PatchStats(model_type=self.__class__.__name__)

    @abstractmethod
    def patch_rms_norm(self) -> int:
        """Patch RMSNorm layers. Returns number of layers patched."""
        pass

    @abstractmethod
    def patch_swiglu(self) -> int:
        """Patch SwiGLU/MLP layers. Returns number of layers patched."""
        pass

    @abstractmethod
    def patch_rope(self) -> int:
        """Patch RoPE layers. Returns number of layers patched."""
        pass

    @abstractmethod
    def patch_attention(self) -> int:
        """Patch attention layers. Returns number of layers patched."""
        pass

    def patch_all(
        self,
        rms_norm: bool = True,
        swiglu: bool = True,
        rope: bool = True,
        attention: bool = False,  # Attention patching is more complex
    ) -> PatchStats:
        """Apply all specified patches.

        Args:
            rms_norm: Patch RMSNorm layers
            swiglu: Patch SwiGLU/MLP layers
            rope: Patch RoPE embeddings
            attention: Patch attention mechanism

        Returns:
            PatchStats with counts of patched layers
        """
        # Store original modules for unpatching
        if self.model_id not in _ORIGINAL_MODULES:
            _ORIGINAL_MODULES[self.model_id] = {}

        if rms_norm:
            self.stats.rms_norm_patched = self.patch_rms_norm()

        if swiglu:
            self.stats.swiglu_patched = self.patch_swiglu()

        if rope:
            self.stats.rope_patched = self.patch_rope()

        if attention:
            self.stats.attention_patched = self.patch_attention()

        self.stats.total_patched = (
            self.stats.rms_norm_patched +
            self.stats.swiglu_patched +
            self.stats.rope_patched +
            self.stats.attention_patched
        )

        logger.info(str(self.stats))
        return self.stats

    def _store_original(self, name: str, module: nn.Module) -> None:
        """Store original module for later unpatching."""
        if self.model_id not in _ORIGINAL_MODULES:
            _ORIGINAL_MODULES[self.model_id] = {}
        _ORIGINAL_MODULES[self.model_id][name] = module

    def _set_module(self, name: str, new_module: nn.Module) -> None:
        """Set a module in the model by its full name."""
        parts = name.split(".")
        parent = self.model

        for part in parts[:-1]:
            parent = getattr(parent, part)

        setattr(parent, parts[-1], new_module)

    def _get_module(self, name: str) -> nn.Module:
        """Get a module from the model by its full name."""
        parts = name.split(".")
        module = self.model

        for part in parts:
            module = getattr(module, part)

        return module


def register_patcher(patcher_class: Type[ModelPatcher]) -> Type[ModelPatcher]:
    """Decorator to register a model patcher."""
    for model_name in patcher_class.SUPPORTED_MODELS:
        _PATCHERS[model_name.lower()] = patcher_class
    return patcher_class


def get_patcher(model: nn.Module) -> Optional[ModelPatcher]:
    """Get the appropriate patcher for a model.

    Args:
        model: The model to patch

    Returns:
        ModelPatcher instance or None if model not supported
    """
    # Try to identify model type
    model_type = getattr(model.config, "model_type", "").lower()

    if model_type in _PATCHERS:
        return _PATCHERS[model_type](model)

    # Try class name matching
    class_name = model.__class__.__name__.lower()
    for key, patcher_class in _PATCHERS.items():
        if key in class_name:
            return patcher_class(model)

    return None


def get_supported_models() -> List[str]:
    """Get list of supported model types."""
    return list(_PATCHERS.keys())


def unpatch_model(model: nn.Module) -> int:
    """Restore original modules in a model.

    Args:
        model: The model to unpatch

    Returns:
        Number of modules restored
    """
    model_id = id(model)

    if model_id not in _ORIGINAL_MODULES:
        logger.warning("Model has no stored original modules")
        return 0

    count = 0
    for name, original_module in _ORIGINAL_MODULES[model_id].items():
        parts = name.split(".")
        parent = model

        for part in parts[:-1]:
            parent = getattr(parent, part)

        setattr(parent, parts[-1], original_module)
        count += 1

    # Clear stored modules
    del _ORIGINAL_MODULES[model_id]

    logger.info(f"Restored {count} original modules")
    return count


def patch_model(
    model: nn.Module,
    rms_norm: bool = True,
    swiglu: bool = True,
    rope: bool = True,
    attention: bool = False,
) -> PatchStats:
    """Auto-detect model type and apply appropriate patches.

    Args:
        model: Model to patch
        rms_norm: Patch RMSNorm layers
        swiglu: Patch SwiGLU/MLP layers
        rope: Patch RoPE embeddings
        attention: Patch attention mechanism

    Returns:
        PatchStats with patching results

    Raises:
        ValueError: If model type is not supported
    """
    patcher = get_patcher(model)

    if patcher is None:
        supported = ", ".join(get_supported_models())
        raise ValueError(
            f"Model type not supported. Supported models: {supported}"
        )

    return patcher.patch_all(
        rms_norm=rms_norm,
        swiglu=swiglu,
        rope=rope,
        attention=attention,
    )
