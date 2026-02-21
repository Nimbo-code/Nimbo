#!/usr/bin/env python3
# Copyright (c) 2025, Nimbo
# Licensed under the Apache License, Version 2.0

"""
Korean Cooking Domain Instruction Fine-tuning Example

This script demonstrates how to fine-tune a language model on Korean cooking
instructions using the Nimbo framework with LoRA.

Usage:
    python examples/instruction_finetuning/train_korean_cooking.py

Requirements:
    - GPU with at least 8GB VRAM (for 1B model with QLoRA)
    - Or 16GB+ VRAM without quantization
"""

import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
from nimbo import (
    Nimbo,
    NimboConfig,
    LoRAConfig,
    TrainingConfig,
    DeviceConfig,
    QuantizationConfig,
    KernelConfig,
    prepare_instruction_dataset,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    # =========================================================================
    # Configuration
    # =========================================================================

    # Model selection (choose based on your GPU memory)
    # For 8GB VRAM: use Llama-3.2-1B with QLoRA
    # For 16GB+ VRAM: use larger models or without quantization
    BASE_MODEL = "meta-llama/Llama-3.2-1B"  # Small but capable model

    # Dataset path
    DATASET_PATH = Path(__file__).parent / "korean_cooking_dataset.jsonl"

    # Output directory
    OUTPUT_DIR = Path(__file__).parent / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("Nimbo Korean Cooking Instruction Fine-tuning")
    logger.info("=" * 60)
    logger.info(f"Base Model: {BASE_MODEL}")
    logger.info(f"Dataset: {DATASET_PATH}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info("=" * 60)

    # =========================================================================
    # Prepare Dataset
    # =========================================================================

    logger.info("Preparing instruction dataset...")

    # Korean cooking instruction template
    # This template is designed for Korean language instruction following
    TEMPLATE = """### 질문:
{instruction}

### 입력:
{input}

### 답변:
{output}"""

    # Load and format dataset
    dataset = prepare_instruction_dataset(
        source=str(DATASET_PATH),
        instruction_field="instruction",
        input_field="input",
        output_field="output",
        template=TEMPLATE,
    )

    logger.info(f"Dataset loaded: {len(dataset)} samples")
    logger.info(f"Sample:\n{dataset[0]['text'][:500]}...")

    # =========================================================================
    # Configure Training
    # =========================================================================

    # Device configuration (auto-detect CUDA/MPS/CPU)
    device_config = DeviceConfig()
    logger.info(f"Using device: {device_config.device}")

    # LoRA configuration - optimized for instruction tuning
    lora_config = LoRAConfig(
        r=16,                    # LoRA rank (higher = more capacity)
        lora_alpha=32,           # LoRA scaling factor
        lora_dropout=0.05,       # Dropout for regularization
        bias="none",             # Don't train biases
        target_modules=None,     # Auto-detect based on model architecture
    )

    # Training configuration
    training_config = TrainingConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=2,      # Adjust based on GPU memory
        gradient_accumulation_steps=4,       # Effective batch size = 2 * 4 = 8
        learning_rate=2e-4,
        warmup_ratio=0.1,
        logging_steps=5,
        save_steps=50,
        eval_strategy="steps",
        eval_steps=25,
        max_length=1024,                     # Korean cooking responses can be long
        gradient_checkpointing=True,         # Save memory
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        train_on_responses_only=True,        # Only train on response tokens (saves compute)
    )

    # Quantization configuration (QLoRA for memory efficiency)
    # Disable if you have enough VRAM
    use_qlora = False  # A100 80GB has enough VRAM
    quantization_config = QuantizationConfig(
        load_in_4bit=use_qlora,
        bnb_4bit_compute_dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ) if use_qlora else QuantizationConfig()

    # Triton kernel configuration (acceleration)
    # Disabled for now due to transformers version compatibility issues
    kernel_config = KernelConfig(
        use_triton_kernels=False,
        patch_rms_norm=False,
        patch_swiglu=False,
        patch_rope=False,
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
        use_flash_attention=False,  # Set True if flash_attn package installed
        use_triton_kernels=False,   # Disable for now due to transformers version compatibility
    )

    # Print model info
    total_params = sum(p.numel() for p in nimbo.model.parameters())
    trainable_params = sum(p.numel() for p in nimbo.model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

    if nimbo.kernel_patch_stats:
        logger.info(f"Triton kernel patches: {nimbo.kernel_patch_stats}")

    # =========================================================================
    # Train
    # =========================================================================

    logger.info("Starting training...")

    try:
        metrics = nimbo.train()

        logger.info("Training completed!")
        logger.info(f"Final metrics: {metrics}")

        # Get loss history
        loss_history = nimbo.get_loss_history()
        if loss_history:
            logger.info(f"Final train loss: {loss_history['train_losses'][-1]:.4f}")
            if loss_history['eval_losses']:
                logger.info(f"Final eval loss: {loss_history['eval_losses'][-1]:.4f}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # =========================================================================
    # Save Model
    # =========================================================================

    logger.info("Saving model...")

    # Save adapter only (faster, smaller)
    adapter_path = nimbo.save(merge=False)
    logger.info(f"Adapter saved to: {adapter_path}")

    # Optionally merge adapter into base model
    # merged_path = nimbo.save(merge=True)
    # logger.info(f"Merged model saved to: {merged_path}")

    # =========================================================================
    # Test Inference
    # =========================================================================

    logger.info("Testing inference...")

    # Load model for inference
    nimbo.load_model(adapter_path, for_inference=True)

    # Test prompts
    test_prompts = [
        "### 질문:\n김치볶음밥 맛있게 만드는 방법을 알려주세요.\n\n### 입력:\n\n\n### 답변:\n",
        "### 질문:\n된장국과 된장찌개의 차이점이 뭔가요?\n\n### 입력:\n\n\n### 답변:\n",
    ]

    for prompt in test_prompts:
        logger.info(f"\nPrompt: {prompt[:50]}...")
        response = nimbo.inference(
            prompt,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
        )
        logger.info(f"Response: {response}")

    # =========================================================================
    # Summary
    # =========================================================================

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Adapter checkpoint: {adapter_path}")
    logger.info(f"Dataset size: {len(dataset)} samples")
    logger.info(f"Epochs: {training_config.num_train_epochs}")

    # Save training config for reproducibility
    config_path = OUTPUT_DIR / "training_config.yaml"
    nimbo.save_config(str(config_path))
    logger.info(f"Config saved to: {config_path}")

    logger.info("=" * 60)

    return nimbo


if __name__ == "__main__":
    main()
