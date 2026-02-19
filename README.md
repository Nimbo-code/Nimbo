# Nimbo

A modern, feature-rich Python framework for fine-tuning language models with LoRA adapters.

Nimbo is a comprehensive rewrite of SmoLoRA with all identified improvements implemented:

- **Priority 1**: CUDA/MPS/CPU auto-detection, `torch.no_grad()`, `use_cache`, fp16/bf16 support
- **Priority 2**: Configuration dataclasses, validation split, callbacks, logging, adapter-only inference
- **Priority 3**: Auto target_modules detection, batch inference, streaming, config files
- **Priority 4**: Gradient checkpointing, Flash Attention 2, torch.compile, QLoRA

## Installation

```bash
# Clone and install
git clone https://github.com/username/nimbo.git
cd nimbo
pip install -e .

# With optional dependencies
pip install -e ".[all]"     # Full installation
pip install -e ".[qlora]"   # QLoRA support
pip install -e ".[flash]"   # Flash Attention
pip install -e ".[dev]"     # Development tools
```

## Quick Start

```python
from nimbo import Nimbo

# Initialize with auto-detected settings
trainer = Nimbo(
    base_model_name="microsoft/phi-2",
    dataset="yelp_review_full",
    text_field="text",
    output_dir="./output",
)

# Train
trainer.train()

# Save merged model
trainer.save()

# Run inference
trainer.load_model("./output/final_merged")
result = trainer.inference("Write a review:")
print(result)
```

## Features

### Configuration Dataclasses

Full control over every aspect of training and inference:

```python
from nimbo import (
    Nimbo,
    LoRAConfig,
    TrainingConfig,
    InferenceConfig,
    QuantizationConfig,
)

lora_config = LoRAConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    # target_modules auto-detected based on model architecture
)

training_config = TrainingConfig(
    learning_rate=1e-4,
    num_train_epochs=3,
    gradient_checkpointing=True,
    early_stopping_patience=3,
)

trainer = Nimbo(
    base_model_name="microsoft/phi-2",
    dataset="your_dataset",
    lora_config=lora_config,
    training_config=training_config,
)
```

### QLoRA for Memory Efficiency

Train larger models with 4-bit quantization:

```python
from nimbo import Nimbo, QuantizationConfig

quant_config = QuantizationConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_quant_type="nf4",
)

trainer = Nimbo(
    base_model_name="mistralai/Mistral-7B-v0.1",
    dataset="your_dataset",
    quantization_config=quant_config,
)
```

### Independent Inference

Standalone inference without trainer:

```python
from nimbo import NimboInference, load_for_inference

# Quick loading
model = load_for_inference(
    model_path="./output/final_merged",
    quantize="4bit",
)

# Single prompt
result = model.generate("Hello, world!")

# Batch inference
results = model.generate(["Prompt 1", "Prompt 2", "Prompt 3"])

# Streaming
for chunk in model.stream("Once upon a time"):
    print(chunk, end="", flush=True)

# Chat
messages = [
    {"role": "user", "content": "What is Python?"},
]
response = model.chat(messages)
```

### Config Files

Save and load configurations:

```python
# Save config
trainer.save_config("config.yaml")

# Load from config
trainer = Nimbo.from_config(
    config_path="config.yaml",
    base_model_name="microsoft/phi-2",
    dataset="your_dataset",
)
```

### Dataset Utilities

Flexible dataset preparation:

```python
from nimbo import prepare_dataset, prepare_instruction_dataset

# From various sources
dataset = prepare_dataset(
    source="./data.jsonl",  # or folder, .csv, .parquet
    text_field="content",
    chunk_size=256,
    deduplicate=True,
    min_length=50,
)

# Instruction-following format
dataset = prepare_instruction_dataset(
    source="./instructions.jsonl",
    instruction_field="instruction",
    input_field="input",
    output_field="output",
)
```

### Callbacks

Monitor and control training:

```python
from nimbo import Nimbo
from nimbo.callbacks import (
    ProgressCallback,
    EarlyStoppingCallback,
    MemoryCallback,
    WandbCallback,
)

trainer = Nimbo(
    base_model_name="microsoft/phi-2",
    dataset="your_dataset",
    callbacks=[
        ProgressCallback(),
        MemoryCallback(),
        WandbCallback(project="my-project"),
    ],
)
```

## Model Architecture Support

Nimbo auto-detects appropriate LoRA target modules for:

| Architecture | Models |
|--------------|--------|
| LLaMA | LLaMA, LLaMA 2, Code Llama |
| Mistral | Mistral, Mixtral |
| Phi | Phi-1, Phi-1.5, Phi-2 |
| GPT-2 | GPT-2, DistilGPT-2 |
| GPT-NeoX | Pythia, GPT-NeoX |
| Falcon | Falcon-7B, Falcon-40B |
| BLOOM | BLOOM, BLOOMZ |
| OPT | OPT |
| Qwen | Qwen, Qwen2 |
| Gemma | Gemma |

## API Reference

### Nimbo

Main training class.

```python
Nimbo(
    base_model_name: str,           # HuggingFace model name
    dataset: str | Dataset = None,   # Dataset name or object
    text_field: str = "text",        # Text field in dataset
    config: NimboConfig = None,      # Complete configuration
    output_dir: str = "./nimbo_output",
    lora_config: LoRAConfig = None,
    training_config: TrainingConfig = None,
    device_config: DeviceConfig = None,
    quantization_config: QuantizationConfig = None,
    use_flash_attention: bool = False,
    auto_precision: bool = True,
    callbacks: list = None,
)

# Methods
trainer.train(resume_from_checkpoint=None) -> dict
trainer.save(merge=True) -> str
trainer.load_model(model_path, for_inference=True) -> tuple
trainer.inference(prompt, config=None) -> str | list
trainer.get_loss_history() -> dict
trainer.save_config(path) -> None
```

### NimboInference

Standalone inference class.

```python
NimboInference(
    model_path: str,
    device_config: DeviceConfig = None,
    inference_config: InferenceConfig = None,
    quantization_config: QuantizationConfig = None,
    adapter_path: str = None,
    use_flash_attention: bool = False,
    compile_model: bool = False,
)

# Methods
model.generate(prompt) -> str | list
model.stream(prompt) -> Generator
model.chat(messages) -> str
model.merge_adapter(output_path) -> None
```

## Development

```bash
# Setup development environment
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/nimbo

# Format code
black src/ tests/
isort src/ tests/

# Type check
mypy src/nimbo
```

## License

MIT License
