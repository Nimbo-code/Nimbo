"""Main Nimbo class for fine-tuning language models with LoRA.

This module implements all Priority 1-4 improvements over the original SmoLoRA.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset, load_dataset
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import SFTTrainer

from .callbacks import (
    EarlyStoppingCallback,
    LossTrackingCallback,
    MemoryCallback,
    NimboCallback,
    ProgressCallback,
)
from .config import (
    DeviceConfig,
    InferenceConfig,
    KernelConfig,
    LoRAConfig,
    NimboConfig,
    QuantizationConfig,
    TrainingConfig,
)

# Configure logging (Priority 2: Proper logging instead of print statements)
logger = logging.getLogger(__name__)


class Nimbo:
    """Nimbo class for fine-tuning language models with LoRA.

    A comprehensive rewrite of SmoLoRA with all Priority 1-4 improvements:
    - P1: CUDA detection, torch.no_grad, use_cache, fp16/bf16, validation
    - P2: Config dataclasses, validation split, callbacks, logging, adapter-only inference
    - P3: Auto target_modules, batch inference, streaming, config files, independent inference
    - P4: Gradient checkpointing, Flash Attention, torch.compile, QLoRA
    """

    # Auto-detected target modules for common model architectures
    # Includes both Attention and MLP layers for comprehensive fine-tuning
    TARGET_MODULE_MAP: Dict[str, List[str]] = {
        # EXAONE 3.5, 4.0 - Attention + MLP
        "exaone": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # LLaMA family - Attention + MLP
        "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # Mistral - Attention + MLP
        "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # Phi - Attention + MLP (uses dense for attention, fc1/fc2 for MLP)
        "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
        # GPT-2 - Attention + MLP
        "gpt2": ["c_attn", "c_proj", "c_fc"],
        # GPT-NeoX - Attention + MLP
        "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        # Falcon - Attention + MLP
        "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        # BLOOM - Attention + MLP
        "bloom": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        # OPT - Attention + MLP
        "opt": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        # Qwen - Attention + MLP
        "qwen": ["c_attn", "c_proj", "w1", "w2"],
        # Qwen2 - Attention + MLP
        "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # Gemma - Attention + MLP
        "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # Gemma2 - Attention + MLP
        "gemma2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }

    def __init__(
        self,
        base_model_name: str,
        dataset: Optional[Union[str, Dataset]] = None,
        text_field: str = "text",
        config: Optional[NimboConfig] = None,
        output_dir: str = "./nimbo_output",
        # Individual config overrides
        lora_config: Optional[LoRAConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        device_config: Optional[DeviceConfig] = None,
        quantization_config: Optional[QuantizationConfig] = None,
        kernel_config: Optional[KernelConfig] = None,
        # Convenience parameters
        use_flash_attention: bool = False,
        use_triton_kernels: bool = True,
        auto_precision: bool = True,
        callbacks: Optional[List[NimboCallback]] = None,
    ):
        """Initialize Nimbo trainer.

        Args:
            base_model_name: HuggingFace model name or path
            dataset: Dataset name, Dataset object, or None for inference-only mode
            text_field: Field containing text in dataset
            config: Complete NimboConfig (overrides individual configs)
            output_dir: Output directory for checkpoints and merged model
            lora_config: LoRA configuration
            training_config: Training configuration
            device_config: Device configuration
            quantization_config: Quantization configuration (QLoRA)
            kernel_config: Triton kernel configuration
            use_flash_attention: Enable Flash Attention 2 (P4)
            use_triton_kernels: Enable Nimbo Triton kernels for acceleration
            auto_precision: Auto-detect best precision (P1)
            callbacks: List of training callbacks (P2)
        """
        self.base_model_name = base_model_name
        self.text_field = text_field
        self.output_dir = output_dir
        self.use_flash_attention = use_flash_attention
        self.use_triton_kernels = use_triton_kernels
        self.kernel_patch_stats = None

        # Resolve configuration
        if config is not None:
            self.config = config
        else:
            self.config = NimboConfig(
                device=device_config or DeviceConfig(),
                lora=lora_config or LoRAConfig(),
                training=training_config or TrainingConfig(output_dir=output_dir),
                quantization=quantization_config or QuantizationConfig(),
                kernels=kernel_config or KernelConfig(use_triton_kernels=use_triton_kernels),
            )

        # Ensure output_dir is consistent
        self.config.training.output_dir = output_dir

        # Auto-detect precision (P1)
        if auto_precision:
            self.config.training.auto_detect_precision()

        # Store callbacks
        self.callbacks = callbacks or []

        # Initialize attributes
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.trainer: Optional[SFTTrainer] = None
        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None
        self.adapter_checkpoint: Optional[str] = None
        self.merged_model_path: Optional[str] = None
        self.loss_history: Optional[LossTrackingCallback] = None

        # Device setup (P1: Proper CUDA/MPS/CPU detection)
        self.device = self.config.device.device
        logger.info(f"Using device: {self.device}")

        # Load model and tokenizer
        self._load_model()

        # Setup dataset if provided
        if dataset is not None:
            self._setup_dataset(dataset)
            self._setup_trainer()

    def _load_model(self) -> None:
        """Load base model and tokenizer with all P1/P4 optimizations."""
        logger.info(f"Loading model: {self.base_model_name}")

        # Prepare model loading kwargs
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
        }

        # Quantization config (P4: QLoRA support)
        bnb_config = self.config.quantization.to_bnb_config()
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
            logger.info(
                f"Using quantization: "
                f"4bit={self.config.quantization.load_in_4bit}, "
                f"8bit={self.config.quantization.load_in_8bit}"
            )

        # Flash Attention 2 (P4)
        if self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Flash Attention 2 enabled")

        # Device mapping
        if self.config.device.device_map:
            model_kwargs["device_map"] = self.config.device.device_map
        elif self.device != "cpu" and bnb_config is None:
            model_kwargs["device_map"] = {"": self.device}

        # Precision for loading (if not quantized)
        if bnb_config is None:
            if self.config.training.bf16:
                model_kwargs["torch_dtype"] = torch.bfloat16
            elif self.config.training.fp16:
                model_kwargs["torch_dtype"] = torch.float16

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            **model_kwargs,
        )

        # Prepare for k-bit training if quantized (P4)
        if bnb_config is not None:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config.training.gradient_checkpointing,
            )

        # Gradient checkpointing (P4)
        if self.config.training.gradient_checkpointing and bnb_config is None:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Disable cache for training (P1)
        self.model.config.use_cache = False

        # Apply Triton kernel patches for acceleration
        if self.config.kernels.use_triton_kernels:
            self._apply_kernel_patches()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

        logger.info("Model and tokenizer loaded successfully")

    def _auto_detect_target_modules(self) -> List[str]:
        """Auto-detect appropriate LoRA target modules based on model architecture.

        Priority 3: Auto target_modules detection.
        """
        if self.model is None:
            return ["q_proj", "v_proj"]  # Default fallback

        model_type = getattr(self.model.config, "model_type", "").lower()

        for key, modules in self.TARGET_MODULE_MAP.items():
            if key in model_type:
                logger.info(f"Auto-detected target modules for {model_type}: {modules}")
                return modules

        # Fallback: try to find attention modules dynamically
        default_modules = ["q_proj", "v_proj"]
        logger.info(f"Using default target modules: {default_modules}")
        return default_modules

    def _apply_kernel_patches(self) -> None:
        """Apply Nimbo Triton kernel patches to the model for acceleration.

        Supported models: EXAONE, LLaMA, Mistral, Phi, Qwen2
        Patches: RMSNorm (up to 11x), SwiGLU (up to 1.6x), RoPE (up to 5.4x)
        """
        try:
            from .kernels import is_triton_available, patch_model, get_supported_models
        except ImportError:
            logger.warning("Nimbo kernels not available. Skipping kernel patches.")
            return

        if not is_triton_available():
            logger.warning("Triton not available. Skipping kernel patches.")
            return

        if self.model is None:
            return

        # Check if model is supported
        model_type = getattr(self.model.config, "model_type", "").lower()
        supported = get_supported_models()

        if not any(s in model_type for s in supported):
            logger.info(
                f"Model type '{model_type}' not supported for kernel patching. "
                f"Supported: {supported}"
            )
            return

        kernel_config = self.config.kernels

        try:
            self.kernel_patch_stats = patch_model(
                self.model,
                rms_norm=kernel_config.patch_rms_norm,
                swiglu=kernel_config.patch_swiglu,
                rope=kernel_config.patch_rope,
                attention=kernel_config.patch_attention,
            )
            logger.info(f"Nimbo kernel patches applied:\n{self.kernel_patch_stats}")
        except Exception as e:
            logger.warning(f"Failed to apply kernel patches: {e}")
            self.kernel_patch_stats = None

    def _setup_dataset(
        self,
        dataset: Union[str, Dataset],
        eval_split: float = 0.1,
    ) -> None:
        """Setup training and evaluation datasets.

        Priority 2: Validation split support.
        """
        # Load dataset
        if isinstance(dataset, Dataset):
            full_dataset = dataset
        else:
            logger.info(f"Loading dataset: {dataset}")
            full_dataset = load_dataset(dataset, split="train")

        # Ensure text field exists
        if self.text_field not in full_dataset.column_names:
            if "text" not in full_dataset.column_names:
                raise ValueError(
                    f"Text field '{self.text_field}' not found in dataset. "
                    f"Available fields: {full_dataset.column_names}"
                )
        else:
            # Map to standard 'text' field
            full_dataset = full_dataset.map(
                lambda ex: {"text": ex[self.text_field]}
            )

        # Split into train/eval (P2: Validation support)
        if eval_split > 0 and self.config.training.eval_strategy != "no":
            split = full_dataset.train_test_split(test_size=eval_split, seed=42)
            self.train_dataset = split["train"]
            self.eval_dataset = split["test"]
            logger.info(
                f"Dataset split: {len(self.train_dataset)} train, "
                f"{len(self.eval_dataset)} eval"
            )
        else:
            self.train_dataset = full_dataset
            self.eval_dataset = None
            logger.info(f"Dataset loaded: {len(self.train_dataset)} samples")

    def _setup_trainer(self) -> None:
        """Setup SFT trainer with LoRA configuration."""
        if self.model is None or self.train_dataset is None:
            raise RuntimeError("Model and dataset must be loaded first")

        # Auto-detect target modules if not specified (P3)
        lora_config = self.config.lora
        if lora_config.target_modules is None:
            lora_config.target_modules = self._auto_detect_target_modules()

        peft_config = lora_config.to_peft_config()
        sft_config = self.config.training.to_sft_config()

        # Build callbacks list
        trainer_callbacks: List[TrainerCallback] = list(self.callbacks)

        # Add loss tracking callback
        self.loss_history = LossTrackingCallback()
        trainer_callbacks.append(self.loss_history)

        # Add early stopping if configured (P2)
        if self.config.training.early_stopping_patience > 0:
            trainer_callbacks.append(
                EarlyStoppingCallback(
                    patience=self.config.training.early_stopping_patience,
                    threshold=self.config.training.early_stopping_threshold,
                    metric=self.config.training.metric_for_best_model,
                    greater_is_better=self.config.training.greater_is_better,
                )
            )

        # Log response-only training status
        if self.config.training.train_on_responses_only:
            logger.info(
                "Response-only fine-tuning enabled via completion_only_loss. "
                "Loss will only be computed on completion/response tokens."
            )

        # Initialize trainer
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            peft_config=peft_config,
            args=sft_config,
            callbacks=trainer_callbacks if trainer_callbacks else None,
        )

        logger.info("Trainer initialized")

    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """Train the model with LoRA fine-tuning.

        Args:
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training metrics
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Provide a dataset.")

        logger.info("Starting training...")

        # Train
        train_result = self.trainer.train(
            resume_from_checkpoint=resume_from_checkpoint
        )

        # Save adapter checkpoint
        self.adapter_checkpoint = os.path.join(self.output_dir, "adapter_checkpoint")
        self.trainer.model.save_pretrained(self.adapter_checkpoint)
        self.tokenizer.save_pretrained(self.adapter_checkpoint)

        logger.info(f"Training complete. Adapter saved to: {self.adapter_checkpoint}")

        return train_result.metrics

    def save(self, merge: bool = True) -> str:
        """Save the fine-tuned model.

        Args:
            merge: If True, merge adapter into base model. If False, save adapter only.

        Returns:
            Path to saved model
        """
        if self.adapter_checkpoint is None:
            raise RuntimeError("No adapter checkpoint. Run train() first.")

        if not merge:
            logger.info(f"Adapter saved at: {self.adapter_checkpoint}")
            return self.adapter_checkpoint

        logger.info("Merging adapter into base model...")

        # Clean up trainer to free memory
        if self.trainer is not None:
            del self.trainer
            self.trainer = None
        if self.model is not None:
            del self.model
            self.model = None

        self._clear_cache()

        # Reload base model
        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if self.config.training.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif self.config.training.fp16:
            model_kwargs["torch_dtype"] = torch.float16

        if self.device != "cpu":
            model_kwargs["device_map"] = {"": self.device}

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            **model_kwargs,
        )

        # Load and merge adapter
        model_with_adapter = PeftModel.from_pretrained(
            base_model,
            self.adapter_checkpoint,
        )
        merged_model = model_with_adapter.merge_and_unload()

        # Save merged model
        self.merged_model_path = os.path.join(self.output_dir, "final_merged")
        merged_model.save_pretrained(self.merged_model_path)
        self.tokenizer.save_pretrained(self.merged_model_path)

        # Clean up
        del merged_model
        del model_with_adapter
        del base_model
        self._clear_cache()

        logger.info(f"Merged model saved to: {self.merged_model_path}")
        return self.merged_model_path

    def load_model(
        self,
        model_path: str,
        for_inference: bool = True,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load a saved model for inference.

        Args:
            model_path: Path to saved model
            for_inference: If True, enable inference optimizations

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model from {model_path}...")

        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}

        if self.device != "cpu":
            model_kwargs["device_map"] = {"": self.device}

        # Precision
        if self.config.training.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif self.config.training.fp16:
            model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
        )

        if for_inference:
            self.model.config.use_cache = True  # P1: Enable cache for inference
            self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        logger.info("Model loaded successfully")
        return self.model, self.tokenizer

    def inference(
        self,
        prompt: Union[str, List[str]],
        config: Optional[InferenceConfig] = None,
        **kwargs: Any,
    ) -> Union[str, List[str]]:
        """Run inference on the model.

        Priority 3: Batch inference support.
        Priority 1: torch.no_grad() and use_cache enabled.

        Args:
            prompt: Single prompt or list of prompts
            config: Inference configuration
            **kwargs: Additional generation kwargs

        Returns:
            Generated text(s)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        config = config or self.config.inference
        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]

        logger.info(f"Running inference on {len(prompts)} prompt(s)...")

        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        # Generate with no_grad (P1)
        generate_kwargs = config.to_generate_kwargs()
        generate_kwargs.update(kwargs)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generate_kwargs,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode
        generated_texts = []
        for i, output in enumerate(outputs):
            input_length = inputs["input_ids"][i].shape[0]
            generated_tokens = output[input_length:]
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(text)

        logger.info("Inference complete")
        return generated_texts if is_batch else generated_texts[0]

    def get_loss_history(self) -> Optional[Dict[str, List]]:
        """Get training loss history.

        Returns:
            Dictionary with steps, train_losses, eval_losses
        """
        if self.loss_history is None:
            return None
        return self.loss_history.get_history()

    def _clear_cache(self) -> None:
        """Clear device cache to free memory."""
        import gc

        gc.collect()

        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

    @classmethod
    def from_config(
        cls,
        config_path: str,
        base_model_name: str,
        dataset: Optional[Union[str, Dataset]] = None,
    ) -> "Nimbo":
        """Create Nimbo instance from config file.

        Priority 3: Config file support.

        Args:
            config_path: Path to YAML or JSON config file
            base_model_name: HuggingFace model name
            dataset: Dataset name or object

        Returns:
            Nimbo instance
        """
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            config = NimboConfig.from_yaml(config_path)
        elif config_path.endswith(".json"):
            config = NimboConfig.from_json(config_path)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")

        return cls(
            base_model_name=base_model_name,
            dataset=dataset,
            config=config,
        )

    def save_config(self, path: str) -> None:
        """Save current configuration to file.

        Args:
            path: Path to save config (.yaml or .json)
        """
        if path.endswith(".yaml") or path.endswith(".yml"):
            self.config.to_yaml(path)
        elif path.endswith(".json"):
            self.config.to_json(path)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")

        logger.info(f"Config saved to: {path}")
