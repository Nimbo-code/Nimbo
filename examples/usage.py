"""Example usage of Nimbo framework.

This script demonstrates the complete workflow:
1. Basic training with default settings
2. Advanced configuration with custom settings
3. QLoRA training for memory efficiency
4. Inference with various options
5. Config file usage
"""

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def basic_usage():
    """Basic usage with minimal configuration."""
    from nimbo import Nimbo

    # Initialize with defaults - auto-detects device and precision
    trainer = Nimbo(
        base_model_name="microsoft/phi-2",  # Or any HuggingFace model
        dataset="yelp_review_full",
        text_field="text",
        output_dir="./output/basic",
    )

    # Train
    metrics = trainer.train()
    print(f"Training metrics: {metrics}")

    # Save merged model
    model_path = trainer.save()
    print(f"Model saved to: {model_path}")

    # Load and run inference
    trainer.load_model(model_path)
    result = trainer.inference("Write a restaurant review:")
    print(f"Generated: {result}")


def advanced_usage():
    """Advanced usage with custom configuration."""
    from nimbo import (
        DeviceConfig,
        InferenceConfig,
        LoRAConfig,
        Nimbo,
        TrainingConfig,
        prepare_dataset,
    )

    # Custom LoRA configuration
    lora_config = LoRAConfig(
        r=16,  # Higher rank for more capacity
        lora_alpha=32,
        lora_dropout=0.05,
        # Auto-detect target modules or specify:
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # Custom training configuration
    training_config = TrainingConfig(
        output_dir="./output/advanced",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=3,
        max_steps=-1,  # Use epochs instead of max_steps
        fp16=False,  # Auto-detected if auto_precision=True
        bf16=True,  # Use bf16 if supported
        gradient_checkpointing=True,  # Save memory
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        early_stopping_patience=3,
    )

    # Prepare custom dataset
    dataset = prepare_dataset(
        source="./my_data.jsonl",  # Or folder of .txt files
        text_field="content",
        chunk_size=256,  # Split into 256-word chunks
        deduplicate=True,
        min_length=50,  # Filter short texts
    )

    # Initialize trainer
    trainer = Nimbo(
        base_model_name="microsoft/phi-2",
        dataset=dataset,
        lora_config=lora_config,
        training_config=training_config,
        use_flash_attention=True,  # Use Flash Attention if available
    )

    # Train with validation
    trainer.train()

    # Get loss history
    history = trainer.get_loss_history()
    print(f"Training steps: {len(history['steps'])}")
    print(f"Final train loss: {history['train_losses'][-1]:.4f}")

    # Save
    trainer.save()


def qlora_usage():
    """QLoRA training for memory-efficient fine-tuning."""
    from nimbo import Nimbo, QuantizationConfig

    # Enable 4-bit quantization
    quant_config = QuantizationConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    trainer = Nimbo(
        base_model_name="mistralai/Mistral-7B-v0.1",
        dataset="timdettmers/openassistant-guanaco",
        text_field="text",
        output_dir="./output/qlora",
        quantization_config=quant_config,
    )

    # Train - will use much less GPU memory
    trainer.train()
    trainer.save()


def inference_examples():
    """Various inference options."""
    from nimbo import InferenceConfig, NimboInference, load_for_inference

    # Method 1: Quick loading with convenience function
    model = load_for_inference(
        model_path="./output/basic/final_merged",
        device="cuda",  # or "mps" or "cpu"
        quantize="4bit",  # Optional: quantize for inference
    )

    # Single inference
    result = model.generate("Hello, how are you?")
    print(result)

    # Batch inference
    prompts = [
        "Write a poem about coding:",
        "Explain quantum computing:",
        "Tell me a joke:",
    ]
    results = model.generate(prompts)
    for prompt, result in zip(prompts, results):
        print(f"\nPrompt: {prompt}")
        print(f"Response: {result}")

    # Streaming inference
    print("\nStreaming response:")
    for chunk in model.stream("Once upon a time"):
        print(chunk, end="", flush=True)
    print()

    # Chat-style inference
    messages = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "What can I do with it?"},
    ]
    response = model.chat(messages)
    print(f"\nChat response: {response}")

    # Method 2: Full control with NimboInference
    inference_config = InferenceConfig(
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
    )

    model = NimboInference(
        model_path="./output/basic/final_merged",
        inference_config=inference_config,
        use_flash_attention=True,
        compile_model=True,  # Use torch.compile for speed
    )

    result = model("Explain the theory of relativity:", config=inference_config)
    print(result)


def adapter_only_inference():
    """Load adapter without merging for flexibility."""
    from nimbo import NimboInference

    # Load base model with adapter
    model = NimboInference(
        model_path="microsoft/phi-2",  # Base model
        adapter_path="./output/basic/adapter_checkpoint",  # LoRA adapter
    )

    result = model.generate("Hello world!")
    print(result)

    # Optionally merge and save
    model.merge_adapter("./output/merged_later")


def config_file_usage():
    """Using YAML/JSON config files."""
    from nimbo import Nimbo, NimboConfig

    # Save current config
    trainer = Nimbo(
        base_model_name="microsoft/phi-2",
        dataset="yelp_review_full",
        output_dir="./output/config_test",
    )
    trainer.save_config("./config.yaml")

    # Later: load from config
    trainer = Nimbo.from_config(
        config_path="./config.yaml",
        base_model_name="microsoft/phi-2",
        dataset="yelp_review_full",
    )
    trainer.train()


def instruction_tuning():
    """Instruction-following fine-tuning."""
    from nimbo import Nimbo, prepare_instruction_dataset

    # Prepare instruction dataset
    dataset = prepare_instruction_dataset(
        source="./instructions.jsonl",
        instruction_field="instruction",
        input_field="input",
        output_field="output",
        template=(
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n{output}"
        ),
    )

    trainer = Nimbo(
        base_model_name="microsoft/phi-2",
        dataset=dataset,
        output_dir="./output/instruction",
    )

    trainer.train()
    trainer.save()


if __name__ == "__main__":
    import sys

    examples = {
        "basic": basic_usage,
        "advanced": advanced_usage,
        "qlora": qlora_usage,
        "inference": inference_examples,
        "adapter": adapter_only_inference,
        "config": config_file_usage,
        "instruction": instruction_tuning,
    }

    if len(sys.argv) > 1 and sys.argv[1] in examples:
        examples[sys.argv[1]]()
    else:
        print("Available examples:")
        for name in examples:
            print(f"  python usage.py {name}")
        print("\nRunning basic example...")
        basic_usage()
