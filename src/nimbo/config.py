"""Configuration dataclasses for Nimbo - Priority 2 improvement."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class DeviceConfig:
    """Device configuration for model training and inference.

    Priority 1: Proper CUDA/MPS/CPU detection.
    """

    device: Optional[str] = None  # None = auto-detect
    device_map: Optional[str] = None  # For multi-GPU: "auto", "balanced", etc.

    def __post_init__(self) -> None:
        if self.device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration.

    Priority 2: Configurable LoRA hyperparameters instead of hardcoded values.
    Priority 3: Auto target_modules detection.
    """

    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: Optional[List[str]] = None  # None = auto-detect
    modules_to_save: Optional[List[str]] = None

    # Additional PEFT kwargs passthrough
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_peft_config(self) -> "PeftLoraConfig":
        """Convert to PEFT LoraConfig object."""
        from peft import LoraConfig as PeftLoraConfig

        config_dict = {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "task_type": self.task_type,
        }

        if self.target_modules is not None:
            config_dict["target_modules"] = self.target_modules

        if self.modules_to_save is not None:
            config_dict["modules_to_save"] = self.modules_to_save

        config_dict.update(self.extra_kwargs)

        return PeftLoraConfig(**config_dict)


@dataclass
class TrainingConfig:
    """Training configuration for SFT (Supervised Fine-Tuning).

    Priority 1: fp16/bf16 support.
    Priority 2: Validation split, early stopping.
    Priority 4: Gradient checkpointing.
    """

    output_dir: str = "./nimbo_output"

    # Batch settings
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    # Training duration
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 = use num_train_epochs

    # Learning rate
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0

    # Optimizer
    optim: str = "adamw_torch"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Precision - Priority 1: fp16/bf16 support
    fp16: bool = False
    bf16: bool = False
    fp16_full_eval: bool = False
    bf16_full_eval: bool = False

    # Memory optimization - Priority 4: Gradient checkpointing
    gradient_checkpointing: bool = False

    # Sequence length
    max_length: int = 1024

    # Logging
    logging_steps: int = 10
    logging_first_step: bool = True

    # Evaluation - Priority 2: Validation support
    eval_strategy: str = "steps"  # "no", "steps", "epoch"
    eval_steps: int = 100

    # Checkpointing
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Early stopping - Priority 2
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0

    # Misc
    seed: int = 42
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True

    # Additional SFTConfig kwargs passthrough
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def auto_detect_precision(self) -> "TrainingConfig":
        """Auto-detect best precision based on hardware."""
        import torch

        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                self.bf16 = True
            else:
                self.fp16 = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't support fp16/bf16 training well
            pass

        return self

    def to_sft_config(self) -> "SFTConfig":
        """Convert to TRL SFTConfig object."""
        from trl import SFTConfig

        config_dict = {
            "output_dir": self.output_dir,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_train_epochs": self.num_train_epochs,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
            "warmup_steps": self.warmup_steps,
            "optim": self.optim,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "fp16_full_eval": self.fp16_full_eval,
            "bf16_full_eval": self.bf16_full_eval,
            "gradient_checkpointing": self.gradient_checkpointing,
            "max_seq_length": self.max_length,
            "logging_steps": self.logging_steps,
            "logging_first_step": self.logging_first_step,
            "eval_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "seed": self.seed,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "dataset_text_field": "text",
        }

        config_dict.update(self.extra_kwargs)

        return SFTConfig(**config_dict)


@dataclass
class InferenceConfig:
    """Inference configuration for text generation.

    Priority 3: Extended generation parameters.
    """

    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    num_beams: int = 1
    num_return_sequences: int = 1

    # Streaming - Priority 3
    stream: bool = False

    # Cache - Priority 1
    use_cache: bool = True

    def to_generate_kwargs(self) -> Dict[str, Any]:
        """Convert to model.generate() kwargs."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature if self.do_sample else None,
            "top_p": self.top_p if self.do_sample else None,
            "top_k": self.top_k if self.do_sample else None,
            "repetition_penalty": self.repetition_penalty,
            "num_beams": self.num_beams,
            "num_return_sequences": self.num_return_sequences,
            "use_cache": self.use_cache,
        }


@dataclass
class QuantizationConfig:
    """Quantization configuration for memory-efficient training and inference.

    Priority 4: QLoRA support with bitsandbytes.
    """

    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    def to_bnb_config(self) -> Optional["BitsAndBytesConfig"]:
        """Convert to BitsAndBytesConfig if quantization is enabled."""
        if not self.load_in_4bit and not self.load_in_8bit:
            return None

        from transformers import BitsAndBytesConfig
        import torch

        compute_dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        if self.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype_map.get(
                    self.bnb_4bit_compute_dtype, torch.float16
                ),
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            )
        else:
            return BitsAndBytesConfig(load_in_8bit=True)


@dataclass
class NimboConfig:
    """Complete Nimbo configuration combining all sub-configs."""

    device: DeviceConfig = field(default_factory=DeviceConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "NimboConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(
            device=DeviceConfig(**data.get("device", {})),
            lora=LoRAConfig(**data.get("lora", {})),
            training=TrainingConfig(**data.get("training", {})),
            inference=InferenceConfig(**data.get("inference", {})),
            quantization=QuantizationConfig(**data.get("quantization", {})),
        )

    @classmethod
    def from_json(cls, path: str) -> "NimboConfig":
        """Load configuration from JSON file."""
        import json

        with open(path, "r") as f:
            data = json.load(f)

        return cls(
            device=DeviceConfig(**data.get("device", {})),
            lora=LoRAConfig(**data.get("lora", {})),
            training=TrainingConfig(**data.get("training", {})),
            inference=InferenceConfig(**data.get("inference", {})),
            quantization=QuantizationConfig(**data.get("quantization", {})),
        )

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        from dataclasses import asdict

        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        import json
        from dataclasses import asdict

        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
