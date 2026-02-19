"""Nimbo - A modern LoRA fine-tuning framework for language models.

Nimbo is a comprehensive rewrite of SmoLoRA with all Priority 1-4 improvements:
- CUDA/MPS/CPU auto-detection
- Configuration dataclasses for full customization
- Validation split and early stopping
- Batch inference and streaming generation
- QLoRA, Flash Attention, gradient checkpointing
- Independent inference class
- Config file support (YAML/JSON)
"""

__version__ = "0.1.0"

from .callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    LossTrackingCallback,
    MemoryCallback,
    NimboCallback,
    ProgressCallback,
    WandbCallback,
    create_default_callbacks,
)
from .config import (
    DeviceConfig,
    InferenceConfig,
    LoRAConfig,
    NimboConfig,
    QuantizationConfig,
    TrainingConfig,
)
from .core import Nimbo
from .dataset import (
    chunk_by_tokens,
    chunk_texts,
    filter_texts,
    load_text_data,
    prepare_chat_dataset,
    prepare_dataset,
    prepare_instruction_dataset,
    read_csv,
    read_jsonl,
    read_parquet,
    read_txt_folder,
)
from .inference import NimboInference, load_for_inference

__all__ = [
    # Main class
    "Nimbo",
    # Inference
    "NimboInference",
    "load_for_inference",
    # Configuration
    "NimboConfig",
    "DeviceConfig",
    "LoRAConfig",
    "TrainingConfig",
    "InferenceConfig",
    "QuantizationConfig",
    # Callbacks
    "NimboCallback",
    "ProgressCallback",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "MemoryCallback",
    "LossTrackingCallback",
    "WandbCallback",
    "create_default_callbacks",
    # Dataset utilities
    "load_text_data",
    "prepare_dataset",
    "prepare_instruction_dataset",
    "prepare_chat_dataset",
    "read_txt_folder",
    "read_jsonl",
    "read_csv",
    "read_parquet",
    "chunk_texts",
    "chunk_by_tokens",
    "filter_texts",
]
