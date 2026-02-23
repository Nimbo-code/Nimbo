# On-Device Deployment Guide

Fine-tune → Merge → Convert → Compile → Deploy to iPhone

## Prerequisites

- Python 3.9+, macOS with Xcode Command Line Tools
- GPU for fine-tuning (NVIDIA CUDA or Apple Silicon MPS)

```bash
git clone https://github.com/Nimbo-code/Nimbo.git && cd Nimbo
python3 -m venv .venv && source .venv/bin/activate

pip install -e ".[all]"
pip install coremltools safetensors scikit-learn
```

## Step 1: Fine-tune

```python
from nimbo import Nimbo

trainer = Nimbo("meta-llama/Llama-3.2-1B-Instruct", dataset="your_data")
trainer.train()
```

Output: `adapter_checkpoint/` (LoRA adapter)

## Step 2: Merge Adapter

```python
trainer.save()  # merge=True by default
```

Output: `final_merged/` (full model with adapter merged)

## Step 3: Convert to CoreML

```python
from nimbo.export.coreml import convert_hf_to_coreml

result = convert_hf_to_coreml(
    'final_merged',
    'coreml_output',
    lut_bits=8,               # Quantization: 4, 6, or 8
    lut_embeddings_bits=-1,   # Embeddings: float16
    lut_lmhead_bits=8,
    split_model=True,
    num_chunks=2,             # Split decoder into N chunks
)
```

Or via CLI:

```bash
python scripts/convert_to_coreml.py \
    --model final_merged \
    --output coreml_output \
    --lut-bits 8 \
    --split-chunks 2
```

Output: `.mlpackage` files + `meta.yaml` + tokenizer files

## Step 4: Compile CoreML Models

```bash
cd coreml_output

for f in *.mlpackage; do
    xcrun coremlcompiler compile "$f" .
done
```

Converts `.mlpackage` → `.mlmodelc` (compiled, ready for device)

## Step 5: Fix File Names

The app expects `_chunk_01of` format (with underscore before chunk number). If your files use `chunk01of`, rename them:

```bash
# Example: 2-chunk model
mv model_FFN_PF_lut8_chunk01of02.mlmodelc model_FFN_PF_lut8_chunk_01of02.mlmodelc
mv model_FFN_PF_lut8_chunk02of02.mlmodelc model_FFN_PF_lut8_chunk_02of02.mlmodelc
```

## Step 6: Update meta.yaml

The app expects `model_info > parameters` format. Replace the generated `meta.yaml`:

```yaml
model_info:
  version: "1.0"
  parameters:
    model_prefix: merged_model        # Must match file name prefix
    context_length: 512
    batch_size: 64
    num_chunks: 2
    lut_ffn: 8                        # Must match lut_bits used
    lut_lmhead: 8
    lut_embeddings: -1                # -1 = float16
    split_lm_head: 8
    vocab_size: 128256                # Check model's config.json
```

Verify file names match the pattern:
```
{model_prefix}_embeddings.mlmodelc
{model_prefix}_FFN_PF_lut{N}_chunk_01of{NN}.mlmodelc
{model_prefix}_lm_head_lut{N}.mlmodelc
```

## Step 7: Deploy to iPhone

1. Open `SampleApp/NimboChat.xcodeproj` in Xcode
2. Set signing team, select your device, build (`Cmd+R`)
3. Copy the model folder (containing `.mlmodelc` files + `meta.yaml` + `tokenizer.json`) to the app:
   - **Finder**: iPhone > Files tab > NimboChat
   - **App**: Use the model picker's folder import button
4. Select the model in the app and start chatting

## Quick Reference

| lut_bits | Size | Quality | Use Case |
|----------|------|---------|----------|
| 4 | Smallest | Lower | Fast prototyping |
| 6 | Medium | Balanced | General use |
| 8 | Largest | Highest | Best quality |

| num_chunks | When to Use |
|------------|-------------|
| 1 | Small models (< 1B params) |
| 2 | Medium models (1B-3B) |
| 4 | Large models (3B+) |
