<p align="center">
  <img src="assets/nimbo_main_logo.png" alt="Nimbo Logo" width="300"/>
</p>

<h3 align="center">
  Fine-tune LLMs with LoRA — Simple, Fast, Memory-Efficient
</h3>

<p align="center">
  <a href="https://github.com/crinex/Nimbo/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"/>
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.9+-green.svg" alt="Python"/>
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch"/>
  </a>
  <a href="https://huggingface.co/">
    <img src="https://img.shields.io/badge/🤗-Transformers-yellow.svg" alt="HuggingFace"/>
  </a>
</p>

<p align="center">
  <b>QLoRA</b> • <b>Flash Attention 2</b> • <b>Auto Device Detection</b> • <b>Streaming Inference</b>
</p>

---

## ⚡ Why Nimbo?

| Feature | Nimbo | Others |
|---------|:-----:|:------:|
| 🚀 CUDA/MPS/CPU Auto-Detection | ✅ | ❌ |
| 🎯 Auto LoRA Target Modules | ✅ | ❌ |
| 📊 QLoRA (4-bit/8-bit) | ✅ | ✅ |
| ⚡ Flash Attention 2 | ✅ | ✅ |
| 🔄 Batch & Streaming Inference | ✅ | ❌ |
| 📁 YAML/JSON Config Files | ✅ | ❌ |
| 🛑 Early Stopping & Callbacks | ✅ | Manual |
| 🧠 Gradient Checkpointing | ✅ | ✅ |
| 🎨 4-Method Simple API | ✅ | ❌ |

---

## 🚀 Quick Start

```bash
pip install git+https://github.com/crinex/Nimbo.git
```

```python
from nimbo import Nimbo

# That's it! Auto-detects device, precision, and LoRA targets
trainer = Nimbo(
    base_model_name="microsoft/phi-2",
    dataset="yelp_review_full",
    output_dir="./output",
)

trainer.train()      # Fine-tune with LoRA
trainer.save()       # Merge & save model

# Inference
trainer.load_model("./output/final_merged")
print(trainer.inference("Write a review:"))
```

---

## 📦 Installation

<details>
<summary><b>🐧 Linux / WSL</b></summary>

```bash
# Basic installation
pip install git+https://github.com/crinex/Nimbo.git

# With QLoRA support
pip install git+https://github.com/crinex/Nimbo.git
pip install bitsandbytes

# Full installation (all features)
pip install "nimbo[all] @ git+https://github.com/crinex/Nimbo.git"
```

</details>

<details>
<summary><b>🍎 macOS (Apple Silicon)</b></summary>

```bash
pip install git+https://github.com/crinex/Nimbo.git
# MPS backend auto-detected!
```

</details>

<details>
<summary><b>🐳 From Source</b></summary>

```bash
git clone https://github.com/crinex/Nimbo.git
cd Nimbo
pip install -e ".[dev]"
```

</details>

---

## 🎯 Supported Models

Nimbo **auto-detects** the optimal LoRA target modules for each architecture:

| Architecture | Models | Target Modules |
|--------------|--------|----------------|
| **LLaMA** | LLaMA, LLaMA 2/3, Code Llama | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| **Mistral** | Mistral, Mixtral | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| **Phi** | Phi-1, Phi-1.5, Phi-2, Phi-3 | `q_proj`, `k_proj`, `v_proj`, `dense` |
| **Qwen** | Qwen, Qwen2 | `c_attn`, `c_proj` |
| **Gemma** | Gemma, Gemma 2 | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| **GPT-2** | GPT-2, DistilGPT-2 | `c_attn`, `c_proj` |
| **Falcon** | Falcon-7B/40B | `query_key_value`, `dense` |
| **BLOOM** | BLOOM, BLOOMZ | `query_key_value`, `dense` |

---

## 💡 Examples

<details>
<summary><b>🔧 Custom Configuration</b></summary>

```python
from nimbo import Nimbo, LoRAConfig, TrainingConfig

trainer = Nimbo(
    base_model_name="mistralai/Mistral-7B-v0.1",
    dataset="your_dataset",
    lora_config=LoRAConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    ),
    training_config=TrainingConfig(
        learning_rate=1e-4,
        num_train_epochs=3,
        gradient_checkpointing=True,
        early_stopping_patience=3,
    ),
    use_flash_attention=True,
)
```

</details>

<details>
<summary><b>🗜️ QLoRA (4-bit Training)</b></summary>

```python
from nimbo import Nimbo, QuantizationConfig

trainer = Nimbo(
    base_model_name="meta-llama/Llama-2-7b-hf",
    dataset="your_dataset",
    quantization_config=QuantizationConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4",
    ),
)
# Train 7B model on consumer GPU!
```

</details>

<details>
<summary><b>🌊 Streaming Inference</b></summary>

```python
from nimbo import NimboInference

model = NimboInference("./output/final_merged")

# Stream tokens as they're generated
for chunk in model.stream("Once upon a time"):
    print(chunk, end="", flush=True)
```

</details>

<details>
<summary><b>📦 Batch Inference</b></summary>

```python
from nimbo import NimboInference

model = NimboInference("./output/final_merged")

# Process multiple prompts at once
prompts = ["Hello!", "How are you?", "Tell me a joke."]
results = model.generate(prompts)
```

</details>

<details>
<summary><b>📄 Config Files</b></summary>

```yaml
# config.yaml
lora:
  r: 16
  lora_alpha: 32

training:
  learning_rate: 0.0001
  num_train_epochs: 3
  gradient_checkpointing: true

quantization:
  load_in_4bit: true
```

```python
trainer = Nimbo.from_config("config.yaml", "microsoft/phi-2", dataset)
```

</details>

<details>
<summary><b>💬 Chat Inference</b></summary>

```python
from nimbo import NimboInference

model = NimboInference("./output/final_merged")

messages = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."},
    {"role": "user", "content": "What can I do with it?"},
]
response = model.chat(messages)
```

</details>

---

## 📊 API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `Nimbo` | Main trainer class for fine-tuning |
| `NimboInference` | Standalone inference engine |
| `NimboConfig` | Complete configuration container |

### Configuration

| Config | Purpose |
|--------|---------|
| `LoRAConfig` | LoRA hyperparameters (r, alpha, dropout) |
| `TrainingConfig` | Training settings (lr, epochs, batch size) |
| `InferenceConfig` | Generation settings (temperature, top_p) |
| `QuantizationConfig` | QLoRA settings (4-bit, 8-bit) |
| `DeviceConfig` | Device selection (cuda, mps, cpu) |

### Dataset Utilities

| Function | Description |
|----------|-------------|
| `prepare_dataset()` | Load from .txt, .csv, .jsonl, .parquet |
| `prepare_instruction_dataset()` | Alpaca-style instruction format |
| `prepare_chat_dataset()` | Chat/conversation format |

### Callbacks

| Callback | Purpose |
|----------|---------|
| `ProgressCallback` | Training progress logging |
| `EarlyStoppingCallback` | Stop on metric plateau |
| `MemoryCallback` | GPU memory monitoring |
| `LossTrackingCallback` | Loss history recording |
| `WandbCallback` | Weights & Biases integration |

---

## 🛠️ Development

```bash
# Setup development environment
git clone https://github.com/crinex/Nimbo.git
cd Nimbo
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black src/ tests/
isort src/ tests/

# Type check
mypy src/nimbo
```

---

## 🔗 Links

| Resource | Link |
|----------|------|
| 📖 Documentation | Coming Soon |
| 🐛 Issues | [GitHub Issues](https://github.com/crinex/Nimbo/issues) |
| 💬 Discussions | [GitHub Discussions](https://github.com/crinex/Nimbo/discussions) |

---

## 📜 License

[MIT License](LICENSE) - Feel free to use Nimbo for personal and commercial projects!

---

<p align="center">
  Made with ☁️ by the Nimbo Team
</p>
