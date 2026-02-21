#!/usr/bin/env python3
# Copyright (c) 2025, Nimbo
# Licensed under MIT License

"""
Large-scale Korean Instruction Fine-tuning with Nimbo

Uses the heegyu/open-korean-instructions dataset (375K+ samples)
with Attention + MLP LoRA targeting for comprehensive fine-tuning.

Usage:
    python examples/instruction_finetuning/train_korean_large.py
"""

import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
from datasets import load_dataset, Dataset
from nimbo import (
    Nimbo,
    LoRAConfig,
    TrainingConfig,
    DeviceConfig,
    QuantizationConfig,
    KernelConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def prepare_korean_instruction_dataset(num_samples: int = 10000) -> Dataset:
    """Load and prepare the open-korean-instructions dataset."""
    logger.info(f"Loading heegyu/open-korean-instructions (sampling {num_samples})...")

    # Load dataset
    ds = load_dataset("heegyu/open-korean-instructions", split="train")
    logger.info(f"Total available samples: {len(ds)}")

    # Shuffle and sample
    ds = ds.shuffle(seed=42).select(range(min(num_samples, len(ds))))

    # Convert format: <usr> ... <bot> ... -> text field
    # The dataset already has 'text' field with <usr> and <bot> tokens
    # We need to convert to instruction format for SFT
    def format_conversation(example):
        text = example['text']
        # Replace tokens for cleaner format
        text = text.replace('<usr>', '### 질문:\n')
        text = text.replace('<bot>', '\n\n### 답변:\n')
        text = text.replace('<sys>', '### 시스템:\n')
        return {"text": text.strip()}

    ds = ds.map(format_conversation)

    logger.info(f"Prepared {len(ds)} samples")
    logger.info(f"Sample:\n{ds[0]['text'][:500]}...")

    return ds


def main():
    # =========================================================================
    # Configuration
    # =========================================================================

    BASE_MODEL = "meta-llama/Llama-3.2-1B"
    NUM_SAMPLES = 5000  # Use 5000 samples for faster training
    OUTPUT_DIR = Path(__file__).parent / "output_large"
    OUTPUT_DIR.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("Nimbo Large-scale Korean Instruction Fine-tuning")
    logger.info("=" * 60)
    logger.info(f"Base Model: {BASE_MODEL}")
    logger.info(f"Dataset: heegyu/open-korean-instructions ({NUM_SAMPLES} samples)")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info("=" * 60)

    # =========================================================================
    # Prepare Dataset
    # =========================================================================

    dataset = prepare_korean_instruction_dataset(num_samples=NUM_SAMPLES)

    # =========================================================================
    # Configure Training
    # =========================================================================

    device_config = DeviceConfig()
    logger.info(f"Using device: {device_config.device}")

    # LoRA configuration - Now includes MLP layers!
    lora_config = LoRAConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=None,  # Auto-detect: will include Attention + MLP layers
    )

    # Training configuration
    training_config = TrainingConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=5,  # 5 epochs for better learning
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size = 16
        learning_rate=2e-4,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=200,
        eval_strategy="steps",
        eval_steps=100,
        max_length=512,  # Shorter for faster training
        gradient_checkpointing=True,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    )

    # No quantization (A100 80GB has enough memory)
    quantization_config = QuantizationConfig()

    # Enable Triton kernels for performance
    kernel_config = KernelConfig(
        use_triton_kernels=True,
    )

    # =========================================================================
    # Initialize Nimbo and Train
    # =========================================================================

    logger.info("Initializing Nimbo trainer...")

    nimbo = Nimbo(
        base_model_name=BASE_MODEL,
        dataset=dataset,
        text_field="text",
        output_dir=str(OUTPUT_DIR),
        lora_config=lora_config,
        training_config=training_config,
        device_config=device_config,
        quantization_config=quantization_config,
        kernel_config=kernel_config,
        use_flash_attention=True,
        use_triton_kernels=True,
    )

    # Print model info
    total_params = sum(p.numel() for p in nimbo.model.parameters())
    trainable_params = sum(p.numel() for p in nimbo.model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

    # =========================================================================
    # Train
    # =========================================================================

    logger.info("Starting training...")

    try:
        metrics = nimbo.train()
        logger.info("Training completed!")
        logger.info(f"Final metrics: {metrics}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # =========================================================================
    # Save Model
    # =========================================================================

    logger.info("Saving model...")
    adapter_path = nimbo.save(merge=False)
    logger.info(f"Adapter saved to: {adapter_path}")

    # Save config
    config_path = OUTPUT_DIR / "training_config.yaml"
    nimbo.save_config(str(config_path))

    # =========================================================================
    # Summary
    # =========================================================================

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Adapter checkpoint: {adapter_path}")
    logger.info(f"Dataset: heegyu/open-korean-instructions")
    logger.info(f"Samples used: {NUM_SAMPLES}")
    logger.info(f"LoRA targets: Attention + MLP layers")
    logger.info("=" * 60)

    return nimbo


if __name__ == "__main__":
    main()
