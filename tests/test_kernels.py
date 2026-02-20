"""Tests for Nimbo Triton kernels."""

import pytest
import torch
import math
from typing import Tuple

# Skip all tests if Triton is not available
triton_available = False
try:
    import triton
    triton_available = True
except ImportError:
    pass

# Skip if no CUDA
cuda_available = torch.cuda.is_available()

pytestmark = [
    pytest.mark.skipif(not cuda_available, reason="CUDA not available"),
    pytest.mark.skipif(not triton_available, reason="Triton not available"),
]


def pytorch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Reference PyTorch implementation of RMSNorm."""
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return (x_norm * weight).to(x.dtype)


def pytorch_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Reference PyTorch implementation of SwiGLU."""
    return torch.nn.functional.silu(gate) * up


def pytorch_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference PyTorch implementation of RoPE."""
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    # Expand cos/sin for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, head_dim//2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # Need to duplicate cos/sin for full head_dim
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class TestRMSNorm:
    """Test cases for RMSNorm Triton kernel."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    def test_basic_forward(self, device):
        """Test basic forward pass."""
        from nimbo.kernels import triton_rms_norm

        batch, seq_len, hidden = 2, 128, 4096
        x = torch.randn(batch, seq_len, hidden, device=device, dtype=torch.float16)
        weight = torch.ones(hidden, device=device, dtype=torch.float16)

        output, rstd = triton_rms_norm(x, weight, eps=1e-5)

        assert output.shape == x.shape
        assert output.dtype == x.dtype
        assert not torch.isnan(output).any()

    def test_correctness(self, device):
        """Test correctness against PyTorch reference."""
        from nimbo.kernels import triton_rms_norm

        batch, seq_len, hidden = 2, 64, 2048
        x = torch.randn(batch, seq_len, hidden, device=device, dtype=torch.float32)
        weight = torch.randn(hidden, device=device, dtype=torch.float32)

        # Triton implementation
        triton_output, _ = triton_rms_norm(x, weight, eps=1e-5)

        # PyTorch reference
        pytorch_output = pytorch_rms_norm(x, weight, eps=1e-5)

        # Compare
        torch.testing.assert_close(triton_output, pytorch_output, rtol=1e-3, atol=1e-3)

    def test_backward(self, device):
        """Test backward pass."""
        from nimbo.kernels import NimboRMSNorm

        hidden = 2048
        norm = NimboRMSNorm(hidden_size=hidden, eps=1e-5, device=device, dtype=torch.float32)

        x = torch.randn(2, 64, hidden, device=device, dtype=torch.float32, requires_grad=True)
        output = norm(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.isnan(x.grad).any()

    def test_different_sizes(self, device):
        """Test various hidden sizes."""
        from nimbo.kernels import triton_rms_norm

        for hidden in [512, 1024, 2048, 4096, 8192]:
            x = torch.randn(1, 32, hidden, device=device, dtype=torch.float16)
            weight = torch.ones(hidden, device=device, dtype=torch.float16)

            output, _ = triton_rms_norm(x, weight)

            assert output.shape == x.shape
            assert not torch.isnan(output).any()


class TestSwiGLU:
    """Test cases for SwiGLU Triton kernel."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    def test_basic_forward(self, device):
        """Test basic forward pass."""
        from nimbo.kernels import triton_swiglu_forward

        batch_seq, intermediate = 256, 11008  # LLaMA-7B intermediate size
        gate = torch.randn(batch_seq, intermediate, device=device, dtype=torch.float16)
        up = torch.randn(batch_seq, intermediate, device=device, dtype=torch.float16)

        output = triton_swiglu_forward(gate, up)

        assert output.shape == gate.shape
        assert output.dtype == gate.dtype
        assert not torch.isnan(output).any()

    def test_correctness(self, device):
        """Test correctness against PyTorch reference."""
        from nimbo.kernels import triton_swiglu_forward

        batch_seq, intermediate = 128, 4096
        gate = torch.randn(batch_seq, intermediate, device=device, dtype=torch.float32)
        up = torch.randn(batch_seq, intermediate, device=device, dtype=torch.float32)

        # Triton implementation
        triton_output = triton_swiglu_forward(gate.contiguous(), up.contiguous())

        # PyTorch reference
        pytorch_output = pytorch_swiglu(gate, up)

        # Compare
        torch.testing.assert_close(triton_output, pytorch_output, rtol=1e-3, atol=1e-3)

    def test_backward(self, device):
        """Test backward pass."""
        from nimbo.kernels import NimboSwiGLU

        swiglu = NimboSwiGLU()

        gate = torch.randn(64, 2048, device=device, dtype=torch.float32, requires_grad=True)
        up = torch.randn(64, 2048, device=device, dtype=torch.float32, requires_grad=True)

        output = swiglu(gate.contiguous(), up.contiguous())
        loss = output.sum()
        loss.backward()

        assert gate.grad is not None
        assert up.grad is not None
        assert gate.grad.shape == gate.shape
        assert not torch.isnan(gate.grad).any()

    def test_3d_input(self, device):
        """Test with 3D input (batch, seq, intermediate)."""
        from nimbo.kernels import triton_swiglu_forward

        batch, seq, intermediate = 2, 128, 4096
        gate = torch.randn(batch, seq, intermediate, device=device, dtype=torch.float16).contiguous()
        up = torch.randn(batch, seq, intermediate, device=device, dtype=torch.float16).contiguous()

        output = triton_swiglu_forward(gate, up)

        assert output.shape == gate.shape
        assert not torch.isnan(output).any()


class TestRoPE:
    """Test cases for RoPE Triton kernel."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    def test_cache_computation(self, device):
        """Test RoPE cache computation."""
        from nimbo.kernels.triton.rope import compute_rope_cache

        seq_len, head_dim = 2048, 128
        cos, sin = compute_rope_cache(
            seq_len=seq_len,
            head_dim=head_dim,
            base=10000.0,
            device=device,
            rope_type="default",
        )

        assert cos.shape == (seq_len, head_dim // 2)
        assert sin.shape == (seq_len, head_dim // 2)
        assert not torch.isnan(cos).any()
        assert not torch.isnan(sin).any()

    def test_llama3_scaling(self, device):
        """Test LLaMA3-style scaling (used by EXAONE)."""
        from nimbo.kernels.triton.rope import compute_rope_cache

        seq_len, head_dim = 32768, 128
        cos, sin = compute_rope_cache(
            seq_len=seq_len,
            head_dim=head_dim,
            base=1000000.0,  # EXAONE base
            device=device,
            scaling_factor=8.0,
            rope_type="llama3",
            original_max_position=8192,
            high_freq_factor=4.0,
            low_freq_factor=1.0,
        )

        assert cos.shape == (seq_len, head_dim // 2)
        assert sin.shape == (seq_len, head_dim // 2)
        assert not torch.isnan(cos).any()

    def test_basic_forward(self, device):
        """Test basic forward pass."""
        from nimbo.kernels import NimboRoPE

        batch, seq_len, num_heads, head_dim = 2, 128, 32, 128
        num_kv_heads = 8  # GQA

        rope = NimboRoPE(
            head_dim=head_dim,
            max_seq_len=2048,
            base=10000.0,
            rope_type="default",
            device=device,
        )

        q = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
        k = torch.randn(batch, seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float32)

        q_out, k_out = rope(q, k)

        assert q_out.shape == q.shape
        assert k_out.shape == k.shape
        assert not torch.isnan(q_out).any()
        assert not torch.isnan(k_out).any()

    def test_exaone_config(self, device):
        """Test EXAONE-specific RoPE configuration."""
        from nimbo.kernels import NimboRoPE

        batch, seq_len, num_heads, head_dim = 1, 512, 32, 128
        num_kv_heads = 8  # EXAONE uses 4:1 GQA

        rope = NimboRoPE(
            head_dim=head_dim,
            max_seq_len=32768,
            base=1000000.0,  # EXAONE uses 1M base
            scaling_factor=8.0,
            rope_type="llama3",
            original_max_position=8192,
            high_freq_factor=4.0,
            low_freq_factor=1.0,
            device=device,
        )

        q = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
        k = torch.randn(batch, seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float32)

        q_out, k_out = rope(q, k)

        assert q_out.shape == q.shape
        assert k_out.shape == k.shape
        assert not torch.isnan(q_out).any()


class TestModelPatcher:
    """Test cases for model patchers."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    def test_get_supported_models(self):
        """Test getting list of supported models."""
        from nimbo.kernels import get_supported_models

        supported = get_supported_models()

        assert isinstance(supported, list)
        assert "llama" in supported
        assert "exaone" in supported

    def test_patch_stats_str(self):
        """Test PatchStats string representation."""
        from nimbo.kernels.patches.base import PatchStats

        stats = PatchStats(
            model_type="TestPatcher",
            rms_norm_patched=10,
            swiglu_patched=5,
            rope_patched=5,
            attention_patched=0,
            total_patched=20,
        )

        str_repr = str(stats)
        assert "TestPatcher" in str_repr
        assert "RMSNorm: 10" in str_repr
        assert "Total: 20" in str_repr


class TestKernelIntegration:
    """Integration tests for kernels."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    def test_mlp_forward(self, device):
        """Test simulated MLP forward with SwiGLU."""
        from nimbo.kernels import triton_swiglu_forward

        batch, seq, hidden = 2, 128, 4096
        intermediate = hidden * 4  # Typical ratio

        # Simulate projections
        x = torch.randn(batch, seq, hidden, device=device, dtype=torch.float16)
        gate_proj = torch.randn(hidden, intermediate, device=device, dtype=torch.float16)
        up_proj = torch.randn(hidden, intermediate, device=device, dtype=torch.float16)
        down_proj = torch.randn(intermediate, hidden, device=device, dtype=torch.float16)

        # Forward pass
        x_flat = x.view(-1, hidden)
        gate = x_flat @ gate_proj
        up = x_flat @ up_proj

        # Apply SwiGLU
        hidden_states = triton_swiglu_forward(gate.contiguous(), up.contiguous())

        # Down projection
        output = hidden_states @ down_proj
        output = output.view(batch, seq, hidden)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_full_attention_block(self, device):
        """Test simulated attention block with RMSNorm and RoPE."""
        from nimbo.kernels import triton_rms_norm, NimboRoPE

        batch, seq, hidden = 2, 64, 2048
        num_heads = 32
        head_dim = hidden // num_heads

        x = torch.randn(batch, seq, hidden, device=device, dtype=torch.float32)
        weight = torch.ones(hidden, device=device, dtype=torch.float32)

        # Apply RMSNorm
        normed, _ = triton_rms_norm(x, weight)
        assert normed.shape == x.shape

        # Simulate QKV projection
        qkv_weight = torch.randn(hidden, hidden * 3, device=device, dtype=torch.float32)
        qkv = normed.view(-1, hidden) @ qkv_weight
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for attention
        q = q.view(batch, seq, num_heads, head_dim)
        k = k.view(batch, seq, num_heads, head_dim)

        # Apply RoPE
        rope = NimboRoPE(head_dim=head_dim, max_seq_len=2048, device=device)
        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert not torch.isnan(q_rot).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
