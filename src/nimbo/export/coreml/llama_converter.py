# Copyright (c) 2025, Nimbo
# Licensed under MIT License
# Based on Anemll (https://github.com/Anemll/Anemll) - MIT License

"""
LLaMA to CoreML converter for Apple Neural Engine.

This module provides the LlamaConverter class for converting HuggingFace
LLaMA models to CoreML format optimized for Apple Neural Engine execution.

Features:
- LUT (Look-Up Table) quantization: 4-bit, 6-bit, 8-bit
- Model splitting: embeddings, decoder layers, lm_head separately or monolithic
- Decoder layer chunking for large models
- Prefill and inference mode support
- Batch processing for efficient prefill

Example usage:
    from nimbo.export.coreml import LlamaConverter, LlamaConfig, LlamaForCausalLM

    # Load config and model
    config = LlamaConfig.from_json("path/to/config.json")
    model = LlamaForCausalLM(config)
    model.load_pretrained_weights("path/to/model")

    # Convert to CoreML
    converter = LlamaConverter(
        model=model,
        context_length=512,
        lut_bits=4,
        batch_size=64,
    )
    coreml_model = converter.convert(split_part='monolithic')
    coreml_model.save("output.mlpackage")
"""

import coremltools as ct
import coremltools.optimize as cto
import numpy as np
import torch
import os
import gc
import warnings

try:
    from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
except Exception:
    SklearnConvergenceWarning = None

if SklearnConvergenceWarning is not None:
    warnings.filterwarnings("ignore", category=SklearnConvergenceWarning)
warnings.filterwarnings("ignore", message="Number of distinct clusters .* smaller than n_clusters")

from .base_converter import BaseConverter
from .llama_model import (
    LlamaModel,
    LlamaConfig,
    LlamaForCausalLM,
    TEST_DEVICE,
    MODEL_DTYPE,
    ENABLE_UNIFIED_CACHE,
)
from .metadata import AddMetadata, get_nimbo_version


class LlamaConverter(BaseConverter):
    """Handles LLaMA model conversion to Apple Neural Engine format.

    This converter transforms PyTorch LLaMA models into CoreML format
    optimized for Apple Neural Engine (ANE) execution on iOS/macOS devices.

    Attributes:
        model: The PyTorch LLaMA model to convert
        context_length: Maximum context length for inference
        state_length: KV cache state length
        lut_bits: Number of bits for LUT quantization (4, 6, or 8)
        per_channel: Group size for per-channel quantization
        batch_size: Batch size for prefill mode
        num_chunks: Number of chunks to split the model into
        argmax_in_model: Whether to compute argmax inside the model
    """

    def __init__(
        self,
        model,
        context_length=512,
        state_length=None,
        lut_bits=4,
        per_channel=8,
        batch_size=64,
        num_chunks=1,
        argmax_in_model=False,
        lut_embeddings_bits=None,
        lut_embeddings_per_channel=8,
        lut_lmhead_bits=None,
        lut_lmhead_per_channel=8,
    ):
        """Initialize the LLaMA converter.

        Args:
            model: The PyTorch LLaMA model to convert
            context_length: Maximum context length (default: 512)
            state_length: KV cache state length (default: same as context_length)
            lut_bits: Number of bits for LUT quantization (default: 4)
            per_channel: Group size for per-channel quantization (default: 8)
            batch_size: Batch size for prefill mode (default: 64)
            num_chunks: Number of chunks for model splitting (default: 1)
            argmax_in_model: Compute argmax in model (default: False)
            lut_embeddings_bits: Override LUT bits for embeddings
            lut_embeddings_per_channel: Per-channel for embeddings
            lut_lmhead_bits: Override LUT bits for LM head
            lut_lmhead_per_channel: Per-channel for LM head
        """
        super().__init__(model)
        self.context_length = context_length
        self.state_length = state_length or context_length
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads
        self.lut_bits = lut_bits
        self.per_channel = per_channel
        self.converted_model = None
        self.batch_size = batch_size
        self.num_chunks = num_chunks
        self.argmax_in_model = argmax_in_model
        self.lut_embeddings_bits = lut_embeddings_bits
        self.lut_embeddings_per_channel = lut_embeddings_per_channel
        self.lut_lmhead_bits = lut_lmhead_bits
        self.lut_lmhead_per_channel = lut_lmhead_per_channel

    def convert(self, split_part=None, chunk_idx=None):
        """Convert model to CoreML format with optional splitting.

        Args:
            split_part: Which part(s) of the model to convert:
                       '1' or 'embeddings' - embeddings only
                       '2' or 'decoder' - decoder layers only (inference mode)
                       '2_prefill' or 'decoder_prefill' - decoder layers (prefill mode)
                       '3' or 'lm_head' - LM head only
                       '123' or 'all' - all components as separate files
                       'monolithic' - single file (inference mode)
                       'monolithic_prefill' - single file (prefill mode)
            chunk_idx: For decoder layers, which chunk to convert (0 to num_chunks-1)
                      If None, converts all layers as one model

        Returns:
            ct.models.MLModel or list[ct.models.MLModel]: Converted model(s)
        """
        # Normalize split_part names
        part_aliases = {
            'embeddings': '1',
            'decoder': '2',
            'decoder_prefill': '2_prefill',
            'lm_head': '3',
            'all': '123',
        }
        split_part = part_aliases.get(split_part, split_part)

        valid_parts = ['1', '2', '2_prefill', '3', '123', 'monolithic', 'monolithic_prefill']
        if split_part not in valid_parts:
            raise ValueError(f"split_part must be one of: {valid_parts}")

        self.preprocess()

        if split_part == '1':
            return self.convert_embeddings(self.model)
        elif split_part == '2':
            return self.convert_decoder(self.model, chunk_idx=chunk_idx)
        elif split_part == '2_prefill':
            return self.convert_decoder_prefill(self.model, chunk_idx=chunk_idx)
        elif split_part == '3':
            lmhead_bits = self.lut_lmhead_bits if self.lut_lmhead_bits is not None else self.lut_bits
            return self.convert_lm_head(
                self.model,
                lut_bits=lmhead_bits,
                argmax_in_model=self.argmax_in_model,
            )
        elif split_part == 'monolithic':
            return self.convert_monolithic(
                self.model,
                is_prefill=False,
                argmax_in_model=self.argmax_in_model,
            )
        elif split_part == 'monolithic_prefill':
            return self.convert_monolithic(
                self.model,
                is_prefill=True,
                argmax_in_model=False,
            )
        elif split_part == '123':
            embeddings_model = self.convert_embeddings(self.model)
            decoder_model = self.convert_decoder(self.model)
            lm_head_model = self.convert_lm_head(
                self.model,
                lut_bits=self.lut_bits,
                argmax_in_model=self.argmax_in_model,
            )
            return [embeddings_model, decoder_model, lm_head_model]

        self.postprocess(num_workers=None)

    def convert_all_chunks(self, output_dir=".", prefix="llama"):
        """Convert all decoder layer chunks separately.

        This is useful for large models that need to be split into
        multiple CoreML models for memory efficiency.

        Args:
            output_dir: Directory to save chunk models
            prefix: Filename prefix

        Returns:
            dict: Paths to all converted models
        """
        os.makedirs(output_dir, exist_ok=True)
        self.preprocess()

        result = {
            'embeddings': None,
            'decoder_chunks': [],
            'decoder_prefill_chunks': [],
            'lm_head': None,
        }

        # Convert embeddings
        print(f"\n{'='*60}")
        print("Converting Embeddings")
        print(f"{'='*60}")
        embeddings = self.convert_embeddings(self.model)
        emb_path = os.path.join(output_dir, f"{prefix}_embeddings.mlpackage")
        if self.lut_bits:
            emb_path = os.path.join(output_dir, f"{prefix}_embeddings_lut{self.lut_bits}.mlpackage")
        embeddings.save(emb_path)
        result['embeddings'] = emb_path

        # Convert decoder chunks
        for chunk_idx in range(self.num_chunks):
            print(f"\n{'='*60}")
            print(f"Converting Decoder Chunk {chunk_idx + 1}/{self.num_chunks}")
            print(f"{'='*60}")

            # Inference mode
            decoder = self.convert_decoder(self.model, chunk_idx=chunk_idx)
            if self.lut_bits:
                dec_path = os.path.join(
                    output_dir,
                    f"{prefix}_decoder_lut{self.lut_bits}_chunk_{chunk_idx+1:02d}of{self.num_chunks:02d}.mlpackage"
                )
            else:
                dec_path = os.path.join(
                    output_dir,
                    f"{prefix}_decoder_chunk_{chunk_idx+1:02d}of{self.num_chunks:02d}.mlpackage"
                )
            decoder.save(dec_path)
            result['decoder_chunks'].append(dec_path)

            # Prefill mode
            prefill = self.convert_decoder_prefill(self.model, chunk_idx=chunk_idx)
            if self.lut_bits:
                pf_path = os.path.join(
                    output_dir,
                    f"{prefix}_decoder_prefill_lut{self.lut_bits}_chunk_{chunk_idx+1:02d}of{self.num_chunks:02d}.mlpackage"
                )
            else:
                pf_path = os.path.join(
                    output_dir,
                    f"{prefix}_decoder_prefill_chunk_{chunk_idx+1:02d}of{self.num_chunks:02d}.mlpackage"
                )
            prefill.save(pf_path)
            result['decoder_prefill_chunks'].append(pf_path)

        # Convert LM head
        print(f"\n{'='*60}")
        print("Converting LM Head")
        print(f"{'='*60}")
        lm_head = self.convert_lm_head(self.model, lut_bits=self.lut_bits, argmax_in_model=self.argmax_in_model)
        if self.lut_bits:
            lm_path = os.path.join(output_dir, f"{prefix}_lm_head_lut{self.lut_bits}.mlpackage")
        else:
            lm_path = os.path.join(output_dir, f"{prefix}_lm_head.mlpackage")
        lm_head.save(lm_path)
        result['lm_head'] = lm_path

        print(f"\n{'='*60}")
        print("Conversion Complete!")
        print(f"{'='*60}")
        print(f"Embeddings: {result['embeddings']}")
        print(f"Decoder chunks: {len(result['decoder_chunks'])}")
        print(f"LM Head: {result['lm_head']}")

        return result

    @staticmethod
    def GetTransformerStates(model, part=None, prefix="model.model."):
        """Get the transformer states for CoreML conversion."""
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        num_layers = model.config.num_hidden_layers

        num_layers_this_part = num_layers * 2
        states = [
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(num_layers_this_part, model.config.num_key_value_heads,
                           model.config.state_length, head_dim),
                    dtype=np.float16
                ),
                name=f"{prefix}kv_cache_0"
            )
        ]

        return states

    def preprocess(self):
        """Preprocessing steps before conversion."""
        print("Preparing model for conversion...")

        print(f"Moving model to device: {TEST_DEVICE}")
        self.model = self.model.to(TEST_DEVICE)
        self.model.eval()

        print("Freezing model parameters...")
        self.model.requires_grad_(False)
        for param in self.model.parameters():
            param.requires_grad = False

        def set_eval_and_freeze(module):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

        self.model.apply(set_eval_and_freeze)
        print("Model preprocessing completed")

    @staticmethod
    def _make_palettizer_config(nbits, per_channel, num_workers):
        """Build an OpPalettizerConfig for LUT quantization."""
        if per_channel <= 0:
            return cto.coreml.OpPalettizerConfig(
                mode="kmeans",
                nbits=nbits,
                granularity="per_tensor",
                num_kmeans_workers=num_workers if num_workers is not None else 1,
            )
        return cto.coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=nbits,
            granularity="per_grouped_channel",
            group_size=per_channel,
            num_kmeans_workers=num_workers if num_workers is not None else 1,
        )

    def postprocess(self, num_workers=None):
        """Apply LUT quantization after conversion."""
        if self.converted_model is not None and self.lut_bits is not None:
            use_per_tensor = self.per_channel <= 0
            if use_per_tensor:
                print(f"Applying LUT quantization with {self.lut_bits} bits (per-tensor)...")
            else:
                print(f"Applying LUT quantization with {self.lut_bits} bits, {self.per_channel} channels per group...")

            try:
                with warnings.catch_warnings():
                    if SklearnConvergenceWarning is not None:
                        warnings.simplefilter('ignore', SklearnConvergenceWarning)
                    warnings.simplefilter('ignore', UserWarning)

                    global_cfg = self._make_palettizer_config(
                        self.lut_bits, self.per_channel, num_workers
                    )

                    op_name_configs = {}
                    has_overrides = (
                        self.lut_embeddings_bits is not None
                        or self.lut_lmhead_bits is not None
                    )

                    if has_overrides:
                        prog = self.converted_model._mil_program
                        for fn_name in prog.functions:
                            fn = prog.functions[fn_name]
                            for op in fn.operations:
                                op_name = op.name or ""
                                if self.lut_embeddings_bits is not None and "embed_tokens" in op_name:
                                    op_name_configs[op_name] = self._make_palettizer_config(
                                        self.lut_embeddings_bits,
                                        self.lut_embeddings_per_channel,
                                        num_workers,
                                    )
                                elif self.lut_lmhead_bits is not None and "lm_head" in op_name:
                                    op_name_configs[op_name] = self._make_palettizer_config(
                                        self.lut_lmhead_bits,
                                        self.lut_lmhead_per_channel,
                                        num_workers,
                                    )

                    config = cto.coreml.OptimizationConfig(
                        global_config=global_cfg,
                        op_name_configs=op_name_configs if op_name_configs else None,
                    )

                    try:
                        self.converted_model = cto.coreml.palettize_weights(
                            self.converted_model, config
                        )
                        print("LUT quantization completed")
                    except ValueError as e:
                        if "Pool not running" in str(e):
                            print("Warning: Multiprocessing error, retrying with single process...")
                            config.global_config.num_kmeans_workers = 1
                            self.converted_model = cto.coreml.palettize_weights(
                                self.converted_model, config
                            )
                            print("LUT quantization completed (single process)")
                        else:
                            raise
            except Exception as e:
                print(f"Warning: LUT quantization failed: {str(e)}")
                print("Continuing with unquantized model...")

    @staticmethod
    def _reset_kv_cache_buffers(module):
        """Clear mutable KV-cache buffers."""
        with torch.no_grad():
            for name, buffer in module.named_buffers():
                if "kv_cache_" in name:
                    buffer.zero_()

    def convert_embeddings(self, model):
        """Convert embeddings layer to CoreML format."""
        print("\nConverting embeddings layer...")

        class EmbeddingsWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.embed_tokens = model.embed_tokens

            def forward(self, input_ids):
                hidden_states = self.embed_tokens(input_ids)
                return hidden_states.to(MODEL_DTYPE)

        wrapper = EmbeddingsWrapper(model)
        wrapper.eval()

        sample_input = torch.zeros((1, 1), dtype=torch.int32, device=TEST_DEVICE)

        print("Tracing embeddings model...")
        traced_model = torch.jit.trace(wrapper, sample_input)

        input_shape = ct.EnumeratedShapes(
            shapes=[[1, 1], [1, self.batch_size]],
            default=[1, 1]
        )

        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="input_ids",
                    shape=input_shape,
                    dtype=np.int32
                )
            ],
            outputs=[
                ct.TensorType(name="hidden_states", dtype=np.float16)
            ],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram"
        )

        embed_bits = self.lut_embeddings_bits if self.lut_embeddings_bits is not None else self.lut_bits
        if embed_bits is not None and embed_bits > 0:  # -1 means skip (keep float16)
            # Temporarily override lut_bits for postprocess
            orig_lut_bits = self.lut_bits
            orig_per_channel = self.per_channel
            self.lut_bits = embed_bits
            self.per_channel = self.lut_embeddings_per_channel
            self.converted_model = mlmodel
            self.postprocess(num_workers=8)
            mlmodel = self.converted_model
            self.lut_bits = orig_lut_bits
            self.per_channel = orig_per_channel
        elif embed_bits is not None and embed_bits < 0:
            print("Skipping embeddings quantization (float16)")

        return mlmodel

    def convert_lm_head(self, model, lut_bits=None, argmax_in_model=False):
        """Convert LM head layer to CoreML."""
        print("\nConverting LM head layer...")

        class LMHeadWrapper(torch.nn.Module):
            def __init__(self, model, argmax_mode=False):
                super().__init__()
                self.argmax_mode = argmax_mode
                if hasattr(model, 'lm_head8_1'):
                    self.heads = [getattr(model, f'lm_head8_{i}') for i in range(1, 9)]
                    self.split_mode = '8way'
                elif hasattr(model, 'lm_head2_1'):
                    self.heads = [model.lm_head2_1, model.lm_head2_2]
                    self.split_mode = '2way'
                elif hasattr(model, 'lm_head1'):
                    self.head = model.lm_head1
                    self.split_mode = 'single'
                else:
                    self.head = model.lm_head
                    self.split_mode = 'linear'

            def forward(self, hidden_states):
                if self.split_mode != 'linear':
                    hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2)

                if self.split_mode == '8way':
                    logits = [head(hidden_states).squeeze(2).transpose(1, 2) for head in self.heads]
                elif self.split_mode == '2way':
                    logits = [
                        self.heads[0](hidden_states).squeeze(2).transpose(1, 2),
                        self.heads[1](hidden_states).squeeze(2).transpose(1, 2),
                    ]
                elif self.split_mode == 'single':
                    logits = [self.head(hidden_states).squeeze(2).transpose(1, 2)]
                else:
                    logits = [self.head(hidden_states)]

                if self.argmax_mode:
                    all_idx = []
                    all_val = []
                    for chunk_logits in logits:
                        chunk_argmax = torch.argmax(chunk_logits, dim=-1, keepdim=True)
                        chunk_max_val = torch.gather(chunk_logits, -1, chunk_argmax)
                        all_idx.append(chunk_argmax.to(torch.int32))
                        all_val.append(chunk_max_val)
                    argmax_idx = torch.cat(all_idx, dim=-1).squeeze(0).squeeze(0)
                    argmax_val = torch.cat(all_val, dim=-1).squeeze(0).squeeze(0)
                    return (argmax_idx, argmax_val)

                if self.split_mode == '8way':
                    return tuple(logits)
                if self.split_mode == '2way':
                    return logits[0], logits[1]
                return logits[0]

        wrapper = LMHeadWrapper(model, argmax_mode=argmax_in_model)
        wrapper.eval()

        sample_input = torch.zeros(
            (1, 1, model.config.hidden_size),
            dtype=MODEL_DTYPE, device=TEST_DEVICE
        )

        print("Tracing LM head model...")
        traced_model = torch.jit.trace(wrapper, sample_input)

        if argmax_in_model:
            outputs = [
                ct.TensorType(name="argmax_idx", dtype=np.int32),
                ct.TensorType(name="argmax_val", dtype=np.float16),
            ]
        elif wrapper.split_mode == '8way':
            outputs = [ct.TensorType(name=f"logits{i}", dtype=np.float16) for i in range(1, 9)]
        elif wrapper.split_mode == '2way':
            outputs = [
                ct.TensorType(name="logits1", dtype=np.float16),
                ct.TensorType(name="logits2", dtype=np.float16)
            ]
        else:
            outputs = [ct.TensorType(name="logits", dtype=np.float16)]

        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="hidden_states",
                    shape=(1, 1, model.config.hidden_size),
                    dtype=np.float16
                )
            ],
            outputs=outputs,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram"
        )

        if lut_bits is not None:
            print(f"Applying LUT quantization with {lut_bits} bits...")
            try:
                config = cto.coreml.OptimizationConfig(
                    global_config=cto.coreml.OpPalettizerConfig(
                        mode="kmeans",
                        nbits=lut_bits,
                        granularity="per_grouped_channel",
                        group_size=self.per_channel,
                        num_kmeans_workers=8
                    ),
                )
                mlmodel = cto.coreml.palettize_weights(mlmodel, config)
                print("LUT quantization completed")
            except Exception as e:
                print(f"Warning: LUT quantization failed: {str(e)}")

        return mlmodel

    def convert_decoder(self, model, chunk_idx=None):
        """Convert Decoder layers to CoreML format (inference mode).

        Args:
            model: The LLaMA model
            chunk_idx: Which chunk to convert (0 to num_chunks-1), or None for all

        Returns:
            ct.models.MLModel: Converted decoder model
        """
        total_layers = model.config.num_hidden_layers

        if chunk_idx is not None:
            base, rem = divmod(total_layers, self.num_chunks)
            start_layer = chunk_idx * base + min(chunk_idx, rem)
            end_layer = start_layer + base + (1 if chunk_idx < rem else 0)
            print(f"\nConverting Decoder layers {start_layer}-{end_layer-1} (chunk {chunk_idx + 1}/{self.num_chunks})...")
        else:
            start_layer = 0
            end_layer = None
            print(f"\nConverting all {total_layers} Decoder layers...")

        class DecoderWrapper(torch.nn.Module):
            def __init__(self, model, start_layer=0, end_layer=None):
                super().__init__()
                self.model = model
                self.start_layer = start_layer
                self.end_layer = end_layer
                self.states = LlamaConverter.GetTransformerStates(model, part='2', prefix="model.model.")

            def forward(self, hidden_states, position_ids, causal_mask, current_pos):
                return self.model.model(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    causal_mask=causal_mask,
                    current_pos=current_pos,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                    IN_PREFILL=False
                )

        try:
            wrapper = DecoderWrapper(model, start_layer, end_layer)
            wrapper.eval()

            hidden_states = torch.zeros(
                (1, 1, model.config.hidden_size),
                dtype=torch.float16, device=TEST_DEVICE
            )
            position_ids = torch.zeros((1,), dtype=torch.long, device=TEST_DEVICE)
            causal_mask = torch.full(
                (1, 1, 1, self.context_length),
                torch.finfo(MODEL_DTYPE).min,
                dtype=MODEL_DTYPE, device=TEST_DEVICE
            )
            current_pos = torch.tensor([0], dtype=torch.long, device=TEST_DEVICE)

            print("Tracing Decoder model...")
            self._reset_kv_cache_buffers(wrapper)
            traced_model = torch.jit.trace(
                wrapper,
                (hidden_states, position_ids, causal_mask, current_pos)
            )
            self._reset_kv_cache_buffers(wrapper)
            self._reset_kv_cache_buffers(traced_model)

            inputs = [
                ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
                ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
                ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
                ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
            ]

            outputs = [ct.TensorType(name="output_hidden_states", dtype=np.float16)]

            mlmodel = ct.convert(
                traced_model,
                inputs=inputs,
                outputs=outputs,
                states=wrapper.states,
                compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                minimum_deployment_target=ct.target.iOS18,
                convert_to="mlprogram"
            )

            print("Decoder layers conversion completed")

            if self.lut_bits:
                self.converted_model = mlmodel
                self.postprocess(num_workers=None)
                mlmodel = self.converted_model

            return mlmodel

        except Exception as e:
            print(f"Error during Decoder conversion: {str(e)}")
            raise

    # Alias for backward compatibility
    convert_FFN = convert_decoder

    def convert_decoder_prefill(self, model, chunk_idx=None):
        """Convert Decoder layers for prefill mode to CoreML format.

        Args:
            model: The LLaMA model
            chunk_idx: Which chunk to convert (0 to num_chunks-1), or None for all

        Returns:
            ct.models.MLModel: Converted decoder prefill model
        """
        total_layers = model.config.num_hidden_layers

        if chunk_idx is not None:
            base, rem = divmod(total_layers, self.num_chunks)
            start_layer = chunk_idx * base + min(chunk_idx, rem)
            end_layer = start_layer + base + (1 if chunk_idx < rem else 0)
            print(f"\nConverting Decoder prefill layers {start_layer}-{end_layer-1} (chunk {chunk_idx + 1}/{self.num_chunks})...")
        else:
            start_layer = 0
            end_layer = None
            print(f"\nConverting all {total_layers} Decoder layers (prefill mode)...")

        class DecoderPrefillWrapper(torch.nn.Module):
            def __init__(self, model, start_layer=0, end_layer=None):
                super().__init__()
                self.model = model
                self.start_layer = start_layer
                self.end_layer = end_layer
                self.states = LlamaConverter.GetTransformerStates(
                    model, part='2_prefill', prefix="model.model."
                )

            def forward(self, hidden_states, position_ids, causal_mask, current_pos):
                return self.model.model(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    causal_mask=causal_mask,
                    current_pos=current_pos,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                    IN_PREFILL=True
                )

        try:
            wrapper = DecoderPrefillWrapper(model, start_layer, end_layer)
            wrapper.eval()

            hidden_states = torch.zeros(
                (1, self.batch_size, model.config.hidden_size),
                dtype=torch.float16, device=TEST_DEVICE
            )
            position_ids = torch.zeros(
                (self.batch_size,), dtype=torch.long, device=TEST_DEVICE
            )
            causal_mask = torch.full(
                (1, 1, self.batch_size, self.context_length),
                torch.finfo(MODEL_DTYPE).min,
                dtype=MODEL_DTYPE, device=TEST_DEVICE
            )
            current_pos = torch.tensor([0], dtype=torch.long, device=TEST_DEVICE)

            print("Tracing Decoder prefill model...")
            self._reset_kv_cache_buffers(wrapper)
            traced_model = torch.jit.trace(
                wrapper,
                (hidden_states, position_ids, causal_mask, current_pos)
            )
            self._reset_kv_cache_buffers(wrapper)
            self._reset_kv_cache_buffers(traced_model)

            inputs = [
                ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
                ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
                ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
                ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
            ]

            outputs = [ct.TensorType(name="output_hidden_states", dtype=np.float16)]

            mlmodel = ct.convert(
                traced_model,
                inputs=inputs,
                outputs=outputs,
                states=wrapper.states,
                compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                minimum_deployment_target=ct.target.iOS18,
                convert_to="mlprogram"
            )

            print("Prefill mode conversion completed")

            if self.lut_bits:
                self.converted_model = mlmodel
                self.postprocess(num_workers=None)
                mlmodel = self.converted_model

            return mlmodel

        except Exception as e:
            print(f"Error during Decoder prefill conversion: {str(e)}")
            raise

    # Alias for backward compatibility
    convert_prefill = convert_decoder_prefill

    def convert_monolithic(self, model, is_prefill=False, argmax_in_model=False):
        """Convert full model to single CoreML model.

        Args:
            model: The LLaMA model to convert
            is_prefill: If True, convert for prefill mode (batch processing)
                       If False, convert for inference mode (single token)
            argmax_in_model: Whether to compute argmax inside the model

        Returns:
            ct.models.MLModel: Monolithic CoreML model
        """
        mode_str = "prefill" if is_prefill else "inference"
        print(f"\nConverting monolithic model for {mode_str} mode...")

        class MonolithicWrapper(torch.nn.Module):
            def __init__(self, model, context_length, is_prefill, argmax_in_model=False):
                super().__init__()
                self.model = model
                self.context_length = context_length
                self.is_prefill = is_prefill
                self.argmax_in_model = argmax_in_model

                if hasattr(model, "lm_head8_1"):
                    self.lm_head_mode = "8"
                    self.lm_heads = [getattr(model, f"lm_head8_{i}") for i in range(1, 9)]
                elif hasattr(model, "lm_head2_1"):
                    self.lm_head_mode = "2"
                    self.lm_heads = [model.lm_head2_1, model.lm_head2_2]
                elif hasattr(model, "lm_head1"):
                    self.lm_head_mode = "1"
                    self.lm_head = model.lm_head1
                else:
                    self.lm_head_mode = "linear"
                    self.lm_head = model.lm_head

            def forward(self, input_ids, position_ids, causal_mask, current_pos):
                # Embeddings
                hidden_states = self.model.embed_tokens(input_ids)
                hidden_states = hidden_states.to(MODEL_DTYPE)

                # Transformer layers
                hidden_states = self.model.model(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    causal_mask=causal_mask,
                    current_pos=current_pos,
                    start_layer=0,
                    end_layer=None,
                    IN_PREFILL=self.is_prefill,
                )

                # LM Head
                if self.lm_head_mode != "linear":
                    hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2)

                if self.lm_head_mode in ("8", "2"):
                    logits_list = [
                        h(hidden_states).squeeze(2).transpose(1, 2)
                        for h in self.lm_heads
                    ]
                elif self.lm_head_mode == "1":
                    logits_list = [self.lm_head(hidden_states).squeeze(2).transpose(1, 2)]
                else:
                    logits_list = [self.lm_head(hidden_states)]

                if self.argmax_in_model and not self.is_prefill:
                    all_idx = []
                    all_val = []
                    for logits in logits_list:
                        chunk_argmax = torch.argmax(logits, dim=-1, keepdim=True)
                        chunk_max_val = torch.gather(logits, -1, chunk_argmax)
                        all_idx.append(chunk_argmax.to(torch.int32))
                        all_val.append(chunk_max_val)
                    argmax_idx = torch.cat(all_idx, dim=-1).squeeze(0).squeeze(0)
                    argmax_val = torch.cat(all_val, dim=-1).squeeze(0).squeeze(0)
                    return (argmax_idx, argmax_val)

                return tuple(logits_list)

        wrapper = MonolithicWrapper(
            model, self.context_length, is_prefill, argmax_in_model=argmax_in_model
        )
        wrapper.eval()

        for param in wrapper.parameters():
            param.requires_grad = False

        # Prepare inputs
        if is_prefill:
            sample_input_ids = torch.zeros(
                (1, self.batch_size), dtype=torch.int32, device=TEST_DEVICE
            )
            sample_position_ids = torch.zeros(
                (self.batch_size,), dtype=torch.int32, device=TEST_DEVICE
            )
            sample_causal_mask = torch.zeros(
                (1, 1, self.batch_size, self.context_length),
                dtype=torch.float16, device=TEST_DEVICE
            )
        else:
            sample_input_ids = torch.zeros(
                (1, 1), dtype=torch.int32, device=TEST_DEVICE
            )
            sample_position_ids = torch.zeros(
                (1,), dtype=torch.int32, device=TEST_DEVICE
            )
            sample_causal_mask = torch.zeros(
                (1, 1, 1, self.context_length),
                dtype=torch.float16, device=TEST_DEVICE
            )

        sample_current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)

        print(f"Sample inputs ({mode_str} mode):")
        print(f"  input_ids: {sample_input_ids.shape}")
        print(f"  position_ids: {sample_position_ids.shape}")
        print(f"  causal_mask: {sample_causal_mask.shape}")
        print(f"  current_pos: {sample_current_pos.shape}")

        print("Tracing monolithic model...")
        self._reset_kv_cache_buffers(wrapper)
        with torch.no_grad():
            traced = torch.jit.trace(
                wrapper,
                (sample_input_ids, sample_position_ids, sample_causal_mask, sample_current_pos),
            )
        self._reset_kv_cache_buffers(wrapper)
        self._reset_kv_cache_buffers(traced)
        print("Tracing completed!")

        # Define outputs
        if argmax_in_model and not is_prefill:
            outputs = [
                ct.TensorType(name="argmax_idx", dtype=np.int32),
                ct.TensorType(name="argmax_val", dtype=np.float16),
            ]
        elif wrapper.lm_head_mode == "8":
            outputs = [ct.TensorType(name=f"logits{i}", dtype=np.float16) for i in range(1, 9)]
        elif wrapper.lm_head_mode == "2":
            outputs = [
                ct.TensorType(name="logits1", dtype=np.float16),
                ct.TensorType(name="logits2", dtype=np.float16),
            ]
        else:
            outputs = [ct.TensorType(name="logits", dtype=np.float16)]

        print("Starting CoreML conversion...")
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="input_ids", shape=sample_input_ids.shape, dtype=np.int32),
                ct.TensorType(name="position_ids", shape=sample_position_ids.shape, dtype=np.int32),
                ct.TensorType(name="causal_mask", shape=sample_causal_mask.shape, dtype=np.float16),
                ct.TensorType(name="current_pos", shape=sample_current_pos.shape, dtype=np.int32),
            ],
            outputs=outputs,
            states=LlamaConverter.GetTransformerStates(model, part=None, prefix="model.model."),
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        print(f"CoreML conversion for monolithic {mode_str} completed!")

        if self.lut_bits:
            print(f"Applying LUT quantization ({self.lut_bits} bits)...")
            self.converted_model = mlmodel
            self.postprocess(num_workers=8)
            mlmodel = self.converted_model

        return mlmodel


def parse_lut_arg(lut_value):
    """Parse LUT argument that can be 'bits', 'bits,per_channel', or 'bits,0' for per-tensor.

    Args:
        lut_value: String value (e.g., '6', '6,4', '4,0' for per-tensor)

    Returns:
        tuple: (lut_bits, per_channel)
    """
    if lut_value is None:
        return None, 8

    if isinstance(lut_value, int):
        return lut_value, 8

    lut_str = str(lut_value).strip().lower()

    if lut_str in ('none', 'no', 'false', ''):
        return None, 8

    if ',' in lut_str:
        parts = lut_str.split(',')
        if len(parts) != 2:
            raise ValueError(f"Invalid LUT format: {lut_value}")
        try:
            lut_bits = int(parts[0])
            per_channel_str = parts[1].strip().lower()
            if per_channel_str in ('tensor', 't', '0'):
                per_channel = 0
            else:
                per_channel = int(parts[1])
            return lut_bits, per_channel
        except ValueError:
            raise ValueError(f"Invalid LUT format: {lut_value}")
    else:
        try:
            lut_bits = int(lut_str)
            return lut_bits, 8
        except ValueError:
            raise ValueError(f"Invalid LUT bits value: {lut_value}")


def convert_llama_to_coreml(
    model_path,
    output_dir=".",
    prefix="llama",
    context_length=512,
    batch_size=64,
    lut_bits=4,
    per_channel=8,
    split_part="monolithic",
    num_chunks=1,
    argmax_in_model=False,
):
    """High-level function to convert a LLaMA model to CoreML.

    Args:
        model_path: Path to the HuggingFace model directory
        output_dir: Output directory for converted models
        prefix: Filename prefix for output files
        context_length: Maximum context length
        batch_size: Batch size for prefill mode
        lut_bits: LUT quantization bits (4, 6, or 8)
        per_channel: Per-channel group size
        split_part: Part to convert ('monolithic', '1', '2', '3', '123', etc.)
        num_chunks: Number of chunks for model splitting
        argmax_in_model: Compute argmax inside model

    Returns:
        The converted CoreML model(s)
    """
    print(f"\nConverting model from: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Context length: {context_length}")
    print(f"LUT quantization: {lut_bits} bits")

    # Load config
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found at {config_path}")

    config = LlamaConfig.from_json(config_path)
    config.context_length = context_length
    if config.state_length < context_length:
        config.state_length = context_length

    print(f"Loaded model config:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  vocab_size: {config.vocab_size}")

    # Initialize and load model
    model = LlamaForCausalLM(config)
    model.load_pretrained_weights(model_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create converter
    converter = LlamaConverter(
        model=model,
        context_length=context_length,
        lut_bits=lut_bits,
        per_channel=per_channel,
        batch_size=batch_size,
        num_chunks=num_chunks,
        argmax_in_model=argmax_in_model,
    )

    # Convert
    converted_model = converter.convert(split_part=split_part)

    # Save
    if split_part in ['monolithic', 'monolithic_prefill']:
        output_name = f"{prefix}_{split_part}"
        if lut_bits:
            output_name += f"_lut{lut_bits}"
        output_path = os.path.join(output_dir, f"{output_name}.mlpackage")

        AddMetadata(converted_model, {
            'context_length': context_length,
            'batch_size': batch_size if 'prefill' in split_part else None,
            'lut_bits': lut_bits,
            'split_part': split_part,
            'argmax_in_model': argmax_in_model if split_part == 'monolithic' else None,
        })

        print(f"Saving to {output_path}")
        converted_model.save(output_path)

    return converted_model
