<p align="center">
  <img src="assets/nimbo_logo2.png" alt="Nimbo Logo" width="300"/>
</p>

<h3 align="center">
  Lightweight LLM Fine-tuning for On-Device Deployment
</h3>

<p align="center">
  <i>From fine-tuning to edge deployment — the complete, lightweight solution</i>
</p>

<p align="center">
  <a href="https://github.com/crinex/Nimbo/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"/>
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.9+-green.svg" alt="Python"/>
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch"/>
  </a>
  <img src="https://img.shields.io/badge/On--Device-Ready-orange?style=flat" alt="On-Device"/>
</p>

<p align="center">
  <b>Fine-tune</b> → <b>Export</b> → <b>Convert</b> → <b>Deploy</b>
</p>

---

## 🎯 What is Nimbo?

**Nimbo** is a lightweight, end-to-end LLM fine-tuning framework designed specifically for **on-device deployment**.

Unlike heavy frameworks like Transformers or Unsloth that pack hundreds of features, Nimbo focuses on **what you actually need** — nothing more, nothing less.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Fine-tune   │ -> │    Export    │ -> │   Convert    │ -> │    Deploy    │
│   (LoRA)     │    │   (Merge)    │    │ (ONNX/etc)   │    │ (Sample App) │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Why Nimbo?

| Pain Point | Heavy Frameworks | Nimbo |
|------------|------------------|-------|
| "I just want to fine-tune and deploy" | 500+ dependencies, complex setup | Minimal deps, just works |
| "My target is mobile/edge devices" | Server-focused, no export tools | On-device first design |
| "I need a working demo fast" | DIY everything | Sample apps included |
| "Training is too slow" | Generic implementation | Triton-optimized kernels |

---

## 🚀 End-to-End Pipeline

### Step 1: Fine-tune (3 lines of code)

```python
from nimbo import Nimbo

trainer = Nimbo("microsoft/phi-2", dataset="your_data")
trainer.train()
trainer.save()  # Merged model ready
```

### Step 2: Export for On-Device Deployment

```python
# Export to CoreML for iOS/macOS (Apple Neural Engine optimized)
from nimbo.export import LlamaConverter, LlamaConfig, LlamaForCausalLM

# Load and convert model
config = LlamaConfig.from_json("./model/config.json")
model = LlamaForCausalLM(config)
model.load_pretrained_weights("./model")

# Convert to CoreML with 4-bit LUT quantization
converter = LlamaConverter(
    model=model,
    context_length=512,
    lut_bits=4,  # 4-bit quantization (supported: 4, 6, 8)
)
coreml_model = converter.convert(split_part="monolithic")
coreml_model.save("model.mlpackage")
```

### Step 3: Deploy with Sample Apps

```bash
# Coming soon: Ready-to-use sample applications
nimbo deploy --format ios --model ./output/model.mlmodel
nimbo deploy --format onnx --model ./output/model.onnx
```

---

## ⚡ Performance: Triton Kernel Acceleration

Nimbo includes **custom Triton GPU kernels** for up to **8x faster** training:

| Kernel | Speedup | Description |
|--------|:-------:|-------------|
| **RMSNorm** | 7-8x | Fused normalization |
| **SwiGLU** | 3-5x | Fused activation |
| **RoPE** | 2x | Fused rotary embeddings |

### Benchmark: EXAONE 4.0 1.2B (A100 80GB)

```
┌─────────────────────────────────────────────────────────────┐
│                  Training Speed Comparison                   │
├─────────────────┬──────────────┬──────────────┬─────────────┤
│ Metric          │ Baseline     │ + Triton     │ Speedup     │
├─────────────────┼──────────────┼──────────────┼─────────────┤
│ Forward Pass    │ 119.79 ms    │ 91.79 ms     │ 1.3x        │
│ Throughput      │ 12,395 tok/s │ 14,116 tok/s │ +14%        │
└─────────────────┴──────────────┴──────────────┴─────────────┘
```

Enable with one line:
```python
from nimbo.kernels import patch_model
patch_model(model)  # 181 layers optimized automatically
```

---

## 🪶 Lightweight by Design

### Nimbo vs. Others

| Feature | Nimbo | Transformers | Unsloth |
|---------|:-----:|:------------:|:-------:|
| Install size | ~50MB | ~500MB+ | ~200MB+ |
| Dependencies | Minimal | 100+ | 50+ |
| On-device export | ✅ | ❌ | ❌ |
| Sample apps | ✅ | ❌ | ❌ |
| Triton kernels | ✅ | ❌ | ✅ |
| Learning curve | 5 min | Hours | 30 min |

### Core Philosophy

- **Essential features only** — No bloat, no unused code
- **On-device first** — Every feature considers edge deployment
- **Zero-to-deploy** — From idea to working app, not just a model file
- **Developer friendly** — Simple API, sensible defaults

---

## 📦 Installation

```bash
# Lightweight install
pip install git+https://github.com/crinex/Nimbo.git

# With all export formats
pip install "nimbo[export] @ git+https://github.com/crinex/Nimbo.git"

# Development
git clone https://github.com/crinex/Nimbo.git && cd Nimbo
pip install -e ".[dev]"
```

---

## 🔧 Environment Setup (Fine-tuning to CoreML)

Complete guide for running the full pipeline: **Fine-tune → Merge → CoreML Convert → Deploy**

### Prerequisites

- **Python** 3.9+ (3.10 recommended)
- **macOS** (CoreML conversion requires macOS with Xcode Command Line Tools)
- **GPU** (optional, for fine-tuning — NVIDIA with CUDA or Apple Silicon MPS)

### Step 1: Create Virtual Environment

```bash
git clone https://github.com/crinex/Nimbo.git && cd Nimbo

python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Core (fine-tuning)
pip install torch transformers datasets peft trl accelerate

# CoreML conversion
pip install coremltools safetensors numpy pyyaml tqdm scikit-learn

# HuggingFace model download (optional)
pip install huggingface_hub

# Install Nimbo itself (editable mode)
pip install -e .
```

Or install everything at once:

```bash
pip install -e ".[all]"
pip install coremltools safetensors scikit-learn
```

### Step 3: Full Pipeline

```python
# 1. Fine-tune
from nimbo import Nimbo

trainer = Nimbo("meta-llama/Llama-3.2-1B-Instruct", dataset="your_data")
trainer.train()
trainer.save()  # Saves merged model to ./final_merged

# 2. Convert to CoreML (split model, per-component quantization)
from nimbo.export.coreml import convert_hf_to_coreml

result = convert_hf_to_coreml(
    'final_merged',
    'coreml_models/output',
    lut_bits=6,               # Decoder: 6-bit LUT
    lut_embeddings_bits=-1,   # Embeddings: float16 (no quantization)
    lut_lmhead_bits=6,        # LM Head: 6-bit LUT
    split_model=True,
    num_chunks=1,
)

# 3. Output: .mlpackage files + meta.yaml + tokenizer files
```

### Step 4: Compile & Deploy to iPhone

```bash
# Compile each .mlpackage to .mlmodelc
xcrun coremlc compile coreml_models/output/model_embeddings.mlpackage coreml_models/output/
xcrun coremlc compile coreml_models/output/model_FFN_PF_lut6.mlpackage coreml_models/output/
xcrun coremlc compile coreml_models/output/model_lm_head_lut6.mlpackage coreml_models/output/

# Transfer to iPhone via Xcode, Finder, or Files app
```

### Package Versions (Tested)

| Package | Version |
|---------|---------|
| torch | 2.7+ |
| transformers | 5.0+ |
| coremltools | 8.2+ |
| safetensors | 0.7+ |
| peft | 0.18+ |
| trl | 0.28+ |

---

## 🎯 Supported Models

### Triton-Optimized Models (Accelerated Training)

| Architecture | Models | Triton Kernels | On-Device |
|--------------|--------|:--------------:|:---------:|
| **LLaMA 3.2** | 1B, 3B Instruct | ✅ Full | ✅ Recommended |
| **EXAONE** | 3.5/4.0 (1.2B-32B) | ✅ Full | ✅ |
| **LLaMA** | 2 (7B-70B), 3 (8B, 70B) | ✅ Full | ✅ |
| **Phi** | Phi-2, Phi-3, Phi-3.5 | ✅ Full | ✅ |
| **Qwen2** | 0.5B, 1.5B, 7B | ✅ Full | ✅ |
| **Mistral** | 7B | ✅ Full | ✅ |

### Other Compatible Models

| Architecture | Models | On-Device |
|--------------|--------|:---------:|
| **Gemma** | Gemma, Gemma 2 | ✅ |
| **Mixtral** | 8x7B | ⚠️ Large |

> **On-Device Recommendation:** LLaMA 3.2 1B/3B, Phi-2, EXAONE 1.2B, Qwen2-1.5B

---

## 💡 Examples

<details>
<summary><b>Basic Fine-tuning</b></summary>

```python
from nimbo import Nimbo

trainer = Nimbo(
    base_model_name="microsoft/phi-2",
    dataset="your_dataset",
    output_dir="./output",
)

trainer.train()
trainer.save()
```

</details>

<details>
<summary><b>QLoRA (4-bit) for Consumer GPUs</b></summary>

```python
from nimbo import Nimbo, QuantizationConfig

trainer = Nimbo(
    base_model_name="meta-llama/Llama-2-7b-hf",
    dataset="your_dataset",
    quantization_config=QuantizationConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
    ),
)
# Fine-tune 7B model on 8GB VRAM!
```

</details>

<details>
<summary><b>Custom Training Config</b></summary>

```python
from nimbo import Nimbo, LoRAConfig, TrainingConfig

trainer = Nimbo(
    base_model_name="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
    dataset="your_dataset",
    lora_config=LoRAConfig(r=16, lora_alpha=32),
    training_config=TrainingConfig(
        learning_rate=1e-4,
        num_train_epochs=3,
        gradient_checkpointing=True,
    ),
)
```

</details>

<details>
<summary><b>OLoRA (Orthogonal LoRA) for Better Stability</b></summary>

```python
from nimbo import Nimbo, LoRAConfig

# OLoRA uses orthogonal initialization via QR decomposition
# Better training stability compared to standard LoRA
trainer = Nimbo(
    base_model_name="microsoft/phi-2",
    dataset="your_dataset",
    lora_config=LoRAConfig(
        r=16,
        lora_alpha=32,
        init_lora_weights="olora",  # Orthogonal initialization
    ),
)

# Other options:
# - init_lora_weights="pissa"  # Principal Singular Values Adaptation
# - init_lora_weights="loftq"  # Quantization-aware initialization
# - use_rslora=True            # Rank-Stabilized LoRA (scales alpha by sqrt(r))
# - use_dora=True              # Weight-Decomposed LoRA
```

</details>

<details>
<summary><b>Response-Only Fine-tuning (Instruction Tuning)</b></summary>

```python
from nimbo import Nimbo, TrainingConfig

# Only compute loss on response/completion tokens
# Instruction/input tokens are masked (labels=-100)
trainer = Nimbo(
    base_model_name="microsoft/phi-2",
    dataset="your_instruction_dataset",  # prompt-completion format
    training_config=TrainingConfig(
        train_on_responses_only=True,  # Only train on completions
        learning_rate=2e-4,
    ),
)

trainer.train()
trainer.save()
```

</details>

<details>
<summary><b>Export to CoreML (iOS/macOS)</b></summary>

```python
from nimbo.export import LlamaConverter, LlamaConfig, LlamaForCausalLM

# Load model configuration and weights
config = LlamaConfig.from_json("./model/config.json")
model = LlamaForCausalLM(config)
model.load_pretrained_weights("./model")

# Create converter with optimizations
converter = LlamaConverter(
    model=model,
    context_length=512,      # Max sequence length
    lut_bits=4,              # LUT quantization (4-bit, 6-bit, or 8-bit)
    batch_size=64,           # Batch size for prefill mode
)

# Convert to monolithic CoreML model
coreml_model = converter.convert(split_part="monolithic")
coreml_model.save("llama_monolithic.mlpackage")

# Or convert as separate components for flexible deployment
embeddings = converter.convert(split_part="1")     # Embeddings
transformer = converter.convert(split_part="2")   # FFN layers
lm_head = converter.convert(split_part="3")       # LM head
```

**Supported split_part options:**
- `"monolithic"` - Single file (inference mode)
- `"monolithic_prefill"` - Single file (prefill mode)
- `"1"` - Embeddings only
- `"2"` - Transformer FFN layers
- `"2_prefill"` - Transformer prefill mode
- `"3"` - LM head only
- `"123"` - All components as separate files

</details>

<details>
<summary><b>Export to ONNX (Coming Soon)</b></summary>

```python
from nimbo import Nimbo

trainer = Nimbo("microsoft/phi-2", dataset="data")
trainer.train()
trainer.save()

# Export for deployment
trainer.export(
    format="onnx",
    output_path="./deploy/model.onnx",
    quantize=True,  # INT8 quantization for edge
)
```

</details>

<details>
<summary><b>Streaming Inference</b></summary>

```python
from nimbo import NimboInference

model = NimboInference("./output/final_merged")

for token in model.stream("Once upon a time"):
    print(token, end="", flush=True)
```

</details>

---

## 🗺️ Roadmap

- [x] LoRA/QLoRA fine-tuning
- [x] OLoRA (Orthogonal LoRA) and advanced variants (RSLoRA, DoRA, PiSSA)
- [x] Response-only fine-tuning (completion_only_loss)
- [x] Triton kernel acceleration
- [x] EXAONE 4.0 optimization
- [x] LLaMA 3.2 (1B, 3B) Triton optimization
- [x] CoreML export for iOS/macOS (ANE optimized, LUT quantization)
- [ ] ONNX export with quantization
- [ ] Sample iOS app (SwiftUI)
- [ ] ONNX Runtime sample app

---

## 📊 API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `Nimbo` | Main trainer for fine-tuning |
| `NimboInference` | Lightweight inference engine |
| `LlamaConverter` | CoreML export for LLaMA models |

### Export Module (`nimbo.export`)

| Class | Description |
|-------|-------------|
| `LlamaConverter` | Convert LLaMA to CoreML (ANE optimized) |
| `LlamaConfig` | Configuration for ANE-optimized model |
| `LlamaForCausalLM` | ANE-optimized LLaMA implementation |
| `BaseConverter` | Abstract base for custom converters |

### Configuration

| Config | Purpose |
|--------|---------|
| `LoRAConfig` | LoRA hyperparameters |
| `TrainingConfig` | Training settings |
| `QuantizationConfig` | QLoRA settings |

### CoreML Export Options

| Option | Description |
|--------|-------------|
| `context_length` | Maximum sequence length (default: 512) |
| `lut_bits` | LUT quantization: 4, 6, or 8 bits |
| `batch_size` | Batch size for prefill mode (default: 64) |
| `split_part` | Model splitting strategy |
| `argmax_in_model` | Compute argmax inside model |

---

## 🛠️ Development

```bash
git clone https://github.com/crinex/Nimbo.git
cd Nimbo
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format
black src/ && isort src/
```

---

## 📜 License

[Apache License 2.0](LICENSE) — Use freely for personal and commercial projects.

---

<p align="center">
  <b>Nimbo</b> — Fine-tune once, deploy everywhere
</p>

<p align="center">
  <sub>Made for developers who ship to production, not just notebooks</sub>
</p>
