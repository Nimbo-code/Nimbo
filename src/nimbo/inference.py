"""Independent inference module for Nimbo - Priority 3 improvement."""

import logging
from typing import Any, Dict, Generator, List, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from .config import DeviceConfig, InferenceConfig, QuantizationConfig

logger = logging.getLogger(__name__)


class NimboInference:
    """Standalone inference class for Nimbo models.

    Priority 3: Independent inference class that doesn't require trainer instance.
    Supports batch inference, streaming, and adapter-only mode.
    """

    def __init__(
        self,
        model_path: str,
        device_config: Optional[DeviceConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
        quantization_config: Optional[QuantizationConfig] = None,
        adapter_path: Optional[str] = None,
        use_flash_attention: bool = False,
        compile_model: bool = False,
    ):
        """Initialize inference engine.

        Args:
            model_path: Path to merged model or base model
            device_config: Device configuration
            inference_config: Inference configuration
            quantization_config: Quantization settings
            adapter_path: Optional path to LoRA adapter (for adapter-only inference)
            use_flash_attention: Enable Flash Attention 2 (Priority 4)
            compile_model: Use torch.compile for optimization (Priority 4)
        """
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.device_config = device_config or DeviceConfig()
        self.inference_config = inference_config or InferenceConfig()
        self.quantization_config = quantization_config or QuantizationConfig()
        self.use_flash_attention = use_flash_attention
        self.compile_model = compile_model

        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None

        self._load_model()

    def _load_model(self) -> None:
        """Load model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}...")

        # Prepare model loading kwargs
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
        }

        # Quantization config (Priority 4)
        bnb_config = self.quantization_config.to_bnb_config()
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
            logger.info("Loading model with quantization")

        # Flash Attention 2 (Priority 4)
        if self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2")

        # Device mapping
        if self.device_config.device_map:
            model_kwargs["device_map"] = self.device_config.device_map
        elif self.device_config.device != "cpu" and bnb_config is None:
            model_kwargs["device_map"] = {"": self.device_config.device}

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_kwargs,
        )

        # Load adapter if specified (Priority 2: adapter-only inference)
        if self.adapter_path:
            from peft import PeftModel

            logger.info(f"Loading LoRA adapter from {self.adapter_path}...")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.adapter_path,
            )

        # Enable cache for inference (Priority 1)
        self.model.config.use_cache = self.inference_config.use_cache

        # Set to eval mode
        self.model.eval()

        # torch.compile optimization (Priority 4)
        if self.compile_model and hasattr(torch, "compile"):
            logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info("Model loaded successfully")

    def generate(
        self,
        prompt: Union[str, List[str]],
        config: Optional[InferenceConfig] = None,
        **kwargs: Any,
    ) -> Union[str, List[str]]:
        """Generate text from prompt(s).

        Priority 3: Supports both single prompt and batch inference.

        Args:
            prompt: Single prompt or list of prompts
            config: Override inference configuration
            **kwargs: Additional generation kwargs

        Returns:
            Generated text(s)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        config = config or self.inference_config
        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]

        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Move to device
        device = self.device_config.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate with no_grad (Priority 1)
        generate_kwargs = config.to_generate_kwargs()
        generate_kwargs.update(kwargs)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generate_kwargs,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode outputs
        generated_texts = []
        for i, output in enumerate(outputs):
            # Remove input tokens from output
            input_length = inputs["input_ids"][i].shape[0]
            generated_tokens = output[input_length:]
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts if is_batch else generated_texts[0]

    def stream(
        self,
        prompt: str,
        config: Optional[InferenceConfig] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Stream generated text token by token.

        Priority 3: Streaming generation support.

        Args:
            prompt: Input prompt
            config: Override inference configuration
            **kwargs: Additional generation kwargs

        Yields:
            Generated text chunks
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded.")

        config = config or self.inference_config

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = self.device_config.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Generate kwargs
        generate_kwargs = config.to_generate_kwargs()
        generate_kwargs.update(kwargs)
        generate_kwargs["streamer"] = streamer

        # Start generation in a thread
        import threading

        def generate_thread() -> None:
            with torch.no_grad():
                self.model.generate(
                    **inputs,
                    **generate_kwargs,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        thread = threading.Thread(target=generate_thread)
        thread.start()

        # Yield tokens as they're generated
        for text in streamer:
            yield text

        thread.join()

    def chat(
        self,
        messages: List[Dict[str, str]],
        config: Optional[InferenceConfig] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Chat-style generation with message history.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            config: Override inference configuration
            system_prompt: Optional system prompt
            **kwargs: Additional generation kwargs

        Returns:
            Generated response
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")

        # Try to use chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            full_messages.extend(messages)

            prompt = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback: simple concatenation
            parts = []
            if system_prompt:
                parts.append(f"System: {system_prompt}\n")
            for msg in messages:
                role = msg["role"].capitalize()
                parts.append(f"{role}: {msg['content']}\n")
            parts.append("Assistant: ")
            prompt = "".join(parts)

        return self.generate(prompt, config, **kwargs)

    def __call__(
        self,
        prompt: Union[str, List[str]],
        **kwargs: Any,
    ) -> Union[str, List[str]]:
        """Shorthand for generate()."""
        return self.generate(prompt, **kwargs)

    def merge_adapter(self, output_path: str) -> None:
        """Merge loaded adapter into base model and save.

        Args:
            output_path: Where to save merged model
        """
        if self.adapter_path is None:
            raise ValueError("No adapter loaded to merge")

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        logger.info("Merging adapter into base model...")
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        logger.info(f"Merged model saved to {output_path}")

    def clear_cache(self) -> None:
        """Clear device cache."""
        device = self.device_config.device
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()


def load_for_inference(
    model_path: str,
    adapter_path: Optional[str] = None,
    device: Optional[str] = None,
    quantize: Optional[str] = None,
    use_flash_attention: bool = False,
) -> NimboInference:
    """Convenience function to load a model for inference.

    Args:
        model_path: Path to model
        adapter_path: Optional adapter path
        device: Device to use (auto-detect if None)
        quantize: "4bit" or "8bit" for quantization
        use_flash_attention: Enable Flash Attention 2

    Returns:
        NimboInference instance
    """
    device_config = DeviceConfig(device=device)

    quantization_config = QuantizationConfig()
    if quantize == "4bit":
        quantization_config.load_in_4bit = True
    elif quantize == "8bit":
        quantization_config.load_in_8bit = True

    return NimboInference(
        model_path=model_path,
        adapter_path=adapter_path,
        device_config=device_config,
        quantization_config=quantization_config,
        use_flash_attention=use_flash_attention,
    )
