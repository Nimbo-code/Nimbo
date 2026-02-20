"""Tests for Nimbo framework."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from nimbo import (
    DeviceConfig,
    InferenceConfig,
    LoRAConfig,
    NimboConfig,
    QuantizationConfig,
    TrainingConfig,
)
from nimbo.dataset import (
    chunk_texts,
    filter_texts,
    prepare_dataset,
    read_csv,
    read_jsonl,
)


class TestDeviceConfig:
    """Tests for DeviceConfig."""

    def test_auto_detect_cpu(self):
        """Test CPU fallback when no GPU available."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                config = DeviceConfig()
                assert config.device == "cpu"

    def test_auto_detect_cuda(self):
        """Test CUDA detection."""
        with patch("torch.cuda.is_available", return_value=True):
            config = DeviceConfig()
            assert config.device == "cuda"

    def test_explicit_device(self):
        """Test explicit device setting."""
        config = DeviceConfig(device="cpu")
        assert config.device == "cpu"


class TestLoRAConfig:
    """Tests for LoRAConfig."""

    def test_default_values(self):
        """Test default LoRA configuration."""
        config = LoRAConfig()
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.1
        assert config.bias == "none"
        assert config.task_type == "CAUSAL_LM"
        assert config.init_lora_weights is True
        assert config.use_rslora is False
        assert config.use_dora is False

    def test_olora_initialization(self):
        """Test OLoRA (orthogonal) initialization setting."""
        config = LoRAConfig(init_lora_weights="olora")
        assert config.init_lora_weights == "olora"

        with patch("peft.LoraConfig") as mock_peft:
            config.to_peft_config()
            call_kwargs = mock_peft.call_args[1]
            assert call_kwargs["init_lora_weights"] == "olora"

    def test_pissa_initialization(self):
        """Test PiSSA initialization setting."""
        config = LoRAConfig(init_lora_weights="pissa")
        assert config.init_lora_weights == "pissa"

    def test_rslora_enabled(self):
        """Test RSLoRA (rank-stabilized) option."""
        config = LoRAConfig(use_rslora=True)
        assert config.use_rslora is True

        with patch("peft.LoraConfig") as mock_peft:
            config.to_peft_config()
            call_kwargs = mock_peft.call_args[1]
            assert call_kwargs["use_rslora"] is True

    def test_dora_enabled(self):
        """Test DoRA (weight-decomposed) option."""
        config = LoRAConfig(use_dora=True)
        assert config.use_dora is True

        with patch("peft.LoraConfig") as mock_peft:
            config.to_peft_config()
            call_kwargs = mock_peft.call_args[1]
            assert call_kwargs["use_dora"] is True

    def test_to_peft_config(self):
        """Test conversion to PEFT config."""
        config = LoRAConfig(r=16, lora_alpha=32)

        with patch("peft.LoraConfig") as mock_peft:
            config.to_peft_config()
            mock_peft.assert_called_once()
            call_kwargs = mock_peft.call_args[1]
            assert call_kwargs["r"] == 16
            assert call_kwargs["lora_alpha"] == 32


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        """Test default training configuration."""
        config = TrainingConfig()
        assert config.per_device_train_batch_size == 1
        assert config.gradient_accumulation_steps == 8
        assert config.learning_rate == 2e-4
        assert config.train_on_responses_only is False

    def test_response_only_training_config(self):
        """Test response-only fine-tuning configuration."""
        config = TrainingConfig(train_on_responses_only=True)
        assert config.train_on_responses_only is True

    def test_response_only_to_sft_config(self):
        """Test response-only training is passed to SFTConfig."""
        config = TrainingConfig(train_on_responses_only=True)

        with patch("trl.SFTConfig") as mock_sft:
            config.to_sft_config()
            call_kwargs = mock_sft.call_args[1]
            assert call_kwargs["completion_only_loss"] is True

    def test_auto_detect_precision_cuda_bf16(self):
        """Test bf16 auto-detection on CUDA."""
        config = TrainingConfig()

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.is_bf16_supported", return_value=True):
                config.auto_detect_precision()
                assert config.bf16 is True
                assert config.fp16 is False

    def test_auto_detect_precision_cuda_fp16(self):
        """Test fp16 fallback when bf16 not supported."""
        config = TrainingConfig()

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.is_bf16_supported", return_value=False):
                config.auto_detect_precision()
                assert config.fp16 is True
                assert config.bf16 is False


class TestInferenceConfig:
    """Tests for InferenceConfig."""

    def test_default_values(self):
        """Test default inference configuration."""
        config = InferenceConfig()
        assert config.max_new_tokens == 256
        assert config.temperature == 0.7
        assert config.use_cache is True

    def test_to_generate_kwargs(self):
        """Test conversion to generate kwargs."""
        config = InferenceConfig(
            max_new_tokens=100,
            temperature=0.5,
            top_p=0.8,
        )
        kwargs = config.to_generate_kwargs()

        assert kwargs["max_new_tokens"] == 100
        assert kwargs["temperature"] == 0.5
        assert kwargs["top_p"] == 0.8
        assert kwargs["use_cache"] is True


class TestQuantizationConfig:
    """Tests for QuantizationConfig."""

    def test_no_quantization(self):
        """Test no quantization returns None."""
        config = QuantizationConfig()
        assert config.to_bnb_config() is None

    def test_4bit_quantization(self):
        """Test 4-bit quantization config."""
        config = QuantizationConfig(load_in_4bit=True)

        with patch("transformers.BitsAndBytesConfig") as mock_bnb:
            config.to_bnb_config()
            mock_bnb.assert_called_once()
            call_kwargs = mock_bnb.call_args[1]
            assert call_kwargs["load_in_4bit"] is True


class TestNimboConfig:
    """Tests for NimboConfig."""

    def test_default_config(self):
        """Test default complete configuration."""
        config = NimboConfig()
        assert isinstance(config.device, DeviceConfig)
        assert isinstance(config.lora, LoRAConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.inference, InferenceConfig)
        assert isinstance(config.quantization, QuantizationConfig)

    def test_yaml_roundtrip(self):
        """Test YAML save and load."""
        config = NimboConfig()
        config.lora.r = 32
        config.training.learning_rate = 1e-5

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config.to_yaml(f.name)
            loaded = NimboConfig.from_yaml(f.name)
            os.unlink(f.name)

        assert loaded.lora.r == 32
        assert loaded.training.learning_rate == 1e-5

    def test_json_roundtrip(self):
        """Test JSON save and load."""
        config = NimboConfig()
        config.lora.lora_alpha = 64

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            config.to_json(f.name)
            loaded = NimboConfig.from_json(f.name)
            os.unlink(f.name)

        assert loaded.lora.lora_alpha == 64


class TestDatasetFunctions:
    """Tests for dataset utility functions."""

    def test_chunk_texts_no_chunking(self):
        """Test no chunking when chunk_size=0."""
        texts = ["hello world", "foo bar"]
        result = chunk_texts(texts, 0)
        assert result == texts

    def test_chunk_texts_with_chunking(self):
        """Test chunking with specified size."""
        texts = ["one two three four five six"]
        result = chunk_texts(texts, 2)
        assert len(result) == 3
        assert result[0] == "one two"
        assert result[1] == "three four"
        assert result[2] == "five six"

    def test_filter_texts_min_length(self):
        """Test filtering by minimum length."""
        texts = ["short", "this is a longer text"]
        result = filter_texts(texts, min_length=10)
        assert len(result) == 1
        assert result[0] == "this is a longer text"

    def test_filter_texts_max_length(self):
        """Test filtering by maximum length."""
        texts = ["short", "this is a longer text"]
        result = filter_texts(texts, max_length=10)
        assert len(result) == 1
        assert result[0] == "short"

    def test_filter_texts_custom_function(self):
        """Test filtering with custom function."""
        texts = ["hello", "HELLO", "world"]
        result = filter_texts(texts, filter_fn=lambda x: x.islower())
        assert result == ["hello", "world"]

    def test_read_jsonl(self):
        """Test reading JSONL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"text": "line 1"}\n')
            f.write('{"text": "line 2"}\n')
            f.name

        result = read_jsonl(f.name, "text")
        os.unlink(f.name)

        assert len(result) == 2
        assert result[0] == "line 1"
        assert result[1] == "line 2"

    def test_read_csv(self):
        """Test reading CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("text,label\n")
            f.write("hello,1\n")
            f.write("world,2\n")
            f.name

        result = read_csv(f.name, "text")
        os.unlink(f.name)

        assert len(result) == 2
        assert result[0] == "hello"
        assert result[1] == "world"

    def test_prepare_dataset_from_list(self):
        """Test preparing dataset from list."""
        texts = ["text 1", "text 2", "text 3"]
        dataset = prepare_dataset(texts)

        assert len(dataset) == 3
        assert "text" in dataset.column_names

    def test_prepare_dataset_deduplication(self):
        """Test deduplication in prepare_dataset."""
        texts = ["text 1", "text 2", "text 1", "text 3"]
        dataset = prepare_dataset(texts, deduplicate=True)

        assert len(dataset) == 3


class TestNimboCore:
    """Tests for Nimbo core class."""

    @pytest.fixture
    def mock_nimbo_deps(self):
        """Mock all Nimbo dependencies."""
        with patch("nimbo.core.AutoModelForCausalLM") as mock_model:
            with patch("nimbo.core.AutoTokenizer") as mock_tokenizer:
                with patch("nimbo.core.SFTTrainer") as mock_trainer:
                    mock_model.from_pretrained.return_value = MagicMock()
                    mock_tokenizer.from_pretrained.return_value = MagicMock()
                    mock_tokenizer.from_pretrained.return_value.pad_token_id = None
                    mock_tokenizer.from_pretrained.return_value.eos_token_id = 0

                    yield {
                        "model": mock_model,
                        "tokenizer": mock_tokenizer,
                        "trainer": mock_trainer,
                    }

    def test_target_module_map(self):
        """Test target module mapping for different architectures."""
        from nimbo import Nimbo

        assert "llama" in Nimbo.TARGET_MODULE_MAP
        assert "gpt2" in Nimbo.TARGET_MODULE_MAP
        assert "phi" in Nimbo.TARGET_MODULE_MAP

    def test_initialization_without_dataset(self, mock_nimbo_deps):
        """Test Nimbo can be initialized without dataset (inference mode)."""
        from nimbo import Nimbo

        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                nimbo = Nimbo(
                    base_model_name="test-model",
                    dataset=None,
                    output_dir="./test_output",
                )

        assert nimbo.model is not None
        assert nimbo.tokenizer is not None
        assert nimbo.trainer is None  # No trainer without dataset


class TestCallbacks:
    """Tests for callback classes."""

    def test_progress_callback(self):
        """Test progress callback."""
        from nimbo.callbacks import ProgressCallback

        callback = ProgressCallback()
        assert callback.name == "ProgressCallback"

    def test_early_stopping_callback(self):
        """Test early stopping callback initialization."""
        from nimbo.callbacks import EarlyStoppingCallback

        callback = EarlyStoppingCallback(patience=5, threshold=0.01)
        assert callback.patience == 5
        assert callback.threshold == 0.01

    def test_loss_tracking_callback(self):
        """Test loss tracking callback."""
        from nimbo.callbacks import LossTrackingCallback

        callback = LossTrackingCallback()
        history = callback.get_history()

        assert "steps" in history
        assert "train_losses" in history
        assert "eval_losses" in history


@pytest.mark.integration
class TestNimboIntegration:
    """Integration tests (require actual model loading)."""

    @pytest.mark.slow
    def test_full_workflow(self):
        """Test complete training workflow with tiny model."""
        # This test requires actual model loading
        # Skip if no GPU available for faster CI
        pytest.skip("Integration test - run manually")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
