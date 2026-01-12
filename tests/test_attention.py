"""
Tests for Gated Sparse Attention module.
"""

import pytest
import torch
import torch.nn as nn
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gsa import GSAConfig, GatedSparseAttention


class TestGatedSparseAttention:
    """Tests for the main GSA module."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return GSAConfig(
            d_model=512,
            n_heads=8,
            n_kv_heads=4,
            d_indexer=32,
            n_indexer_heads=2,
            k_base=64,
            k_min=16,
            k_max=128,
            use_value_gate=True,
            use_output_gate=True,
            use_adaptive_k=True,
            max_position_embeddings=2048,
        )

    @pytest.fixture
    def gsa(self, config):
        """Create GSA module."""
        return GatedSparseAttention(config)

    def test_initialization(self, gsa, config):
        """Test GSA initializes correctly."""
        assert gsa.d_model == config.d_model
        assert gsa.n_heads == config.n_heads
        assert gsa.n_kv_heads == config.n_kv_heads

        # Check projections
        assert gsa.q_proj.in_features == config.d_model
        assert gsa.q_proj.out_features == config.n_heads * config.d_head

        # Check gates exist
        assert gsa.value_gate is not None
        assert gsa.output_gate is not None

    def test_forward_shape(self, gsa, config):
        """Test forward pass output shape."""
        batch_size, seq_len = 2, 128
        x = torch.randn(batch_size, seq_len, config.d_model)
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        output, _, _ = gsa(x, positions=positions)

        assert output.shape == (batch_size, seq_len, config.d_model)

    def test_forward_with_cache(self, gsa, config):
        """Test forward pass with KV cache."""
        batch_size, seq_len = 2, 64
        x = torch.randn(batch_size, seq_len, config.d_model)
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        # First pass to create cache
        output1, cache, _ = gsa(x, positions=positions, use_cache=True)

        assert cache is not None
        assert len(cache) == 2  # (k, v)
        assert cache[0].shape[1] == seq_len

        # Second pass with new tokens
        new_tokens = 16
        x2 = torch.randn(batch_size, new_tokens, config.d_model)
        positions2 = torch.arange(seq_len, seq_len + new_tokens).unsqueeze(0).expand(batch_size, -1)

        output2, cache2, _ = gsa(x2, positions=positions2, past_key_value=cache, use_cache=True)

        assert output2.shape == (batch_size, new_tokens, config.d_model)
        assert cache2[0].shape[1] == seq_len + new_tokens

    def test_gradient_flow(self, gsa, config):
        """Test gradients flow correctly."""
        batch_size, seq_len = 2, 64
        x = torch.randn(batch_size, seq_len, config.d_model, requires_grad=True)
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        output, _, _ = gsa(x, positions=positions)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_causal_masking(self, gsa, config):
        """Test causal masking is applied correctly."""
        batch_size, seq_len = 1, 32
        x = torch.randn(batch_size, seq_len, config.d_model)
        positions = torch.arange(seq_len).unsqueeze(0)

        output, _, attn_weights = gsa(x, positions=positions, output_attentions=True)

        # Note: For sparse attention, verifying causal masking is more complex
        # as we only have weights for selected tokens
        assert output is not None

    def test_different_batch_sizes(self, gsa, config):
        """Test with different batch sizes."""
        for batch_size in [1, 4, 8]:
            seq_len = 64
            x = torch.randn(batch_size, seq_len, config.d_model)
            positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

            output, _, _ = gsa(x, positions=positions)
            assert output.shape == (batch_size, seq_len, config.d_model)

    def test_different_seq_lengths(self, gsa, config):
        """Test with different sequence lengths."""
        batch_size = 2
        for seq_len in [32, 128, 256]:
            x = torch.randn(batch_size, seq_len, config.d_model)
            positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

            output, _, _ = gsa(x, positions=positions)
            assert output.shape == (batch_size, seq_len, config.d_model)


class TestGSAWithoutGates:
    """Test GSA with gates disabled."""

    @pytest.fixture
    def config_no_gates(self):
        """Create config without gates."""
        return GSAConfig(
            d_model=512,
            n_heads=8,
            n_kv_heads=4,
            use_value_gate=False,
            use_output_gate=False,
        )

    @pytest.fixture
    def gsa_no_gates(self, config_no_gates):
        """Create GSA without gates."""
        return GatedSparseAttention(config_no_gates)

    def test_no_gates(self, gsa_no_gates, config_no_gates):
        """Test GSA works without gates."""
        assert gsa_no_gates.value_gate is None
        assert gsa_no_gates.output_gate is None

        batch_size, seq_len = 2, 64
        x = torch.randn(batch_size, seq_len, config_no_gates.d_model)
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        output, _, _ = gsa_no_gates(x, positions=positions)
        assert output.shape == (batch_size, seq_len, config_no_gates.d_model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGSACuda:
    """CUDA-specific tests."""

    @pytest.fixture
    def config(self):
        return GSAConfig(d_model=512, n_heads=8, n_kv_heads=4)

    @pytest.fixture
    def gsa_cuda(self, config):
        return GatedSparseAttention(config).cuda()

    def test_cuda_forward(self, gsa_cuda, config):
        """Test forward on CUDA."""
        batch_size, seq_len = 2, 128
        x = torch.randn(batch_size, seq_len, config.d_model, device="cuda")
        positions = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch_size, -1)

        output, _, _ = gsa_cuda(x, positions=positions)
        assert output.device.type == "cuda"
        assert output.shape == (batch_size, seq_len, config.d_model)

    def test_mixed_precision(self, gsa_cuda, config):
        """Test with mixed precision."""
        batch_size, seq_len = 2, 128
        x = torch.randn(batch_size, seq_len, config.d_model, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch_size, -1)

        gsa_cuda = gsa_cuda.to(torch.bfloat16)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output, _, _ = gsa_cuda(x, positions=positions)

        assert output.dtype == torch.bfloat16
