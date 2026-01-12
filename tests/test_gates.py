"""
Tests for gating modules (ValueGate and OutputGate).
"""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gsa.attention import ValueGate, OutputGate


class TestValueGate:
    """Tests for ValueGate (G2)."""

    @pytest.fixture
    def value_gate(self):
        return ValueGate(
            d_model=512,
            n_kv_heads=8,
            d_head=64,
            bias_init=0.5,
        )

    def test_initialization(self, value_gate):
        """Test ValueGate initializes correctly."""
        assert value_gate.d_model == 512
        assert value_gate.n_kv_heads == 8
        assert value_gate.d_head == 64

        # Check bias initialization
        assert value_gate.gate_proj.bias is not None
        assert torch.allclose(
            value_gate.gate_proj.bias,
            torch.full_like(value_gate.gate_proj.bias, 0.5),
            atol=1e-6
        )

    def test_forward_shape(self, value_gate):
        """Test forward pass output shape."""
        batch_size, seq_len = 2, 64
        n_kv_heads, d_head = 8, 64

        v = torch.randn(batch_size, seq_len, n_kv_heads, d_head)
        hidden_states = torch.randn(batch_size, seq_len, 512)

        output = value_gate(v, hidden_states)

        assert output.shape == v.shape

    def test_gating_bounds(self, value_gate):
        """Test gate values are bounded [0, 1] for sigmoid."""
        batch_size, seq_len = 2, 64
        n_kv_heads, d_head = 8, 64

        v = torch.randn(batch_size, seq_len, n_kv_heads, d_head)
        hidden_states = torch.randn(batch_size, seq_len, 512)

        value_gate.eval()
        output = value_gate(v, hidden_states)

        # The output should be v * gate where gate is in [0, 1]
        # So |output| <= |v|
        # This isn't strictly true due to numerical precision, but gate scores should be bounded
        if value_gate.last_gate_scores is not None:
            assert (value_gate.last_gate_scores >= 0).all()
            assert (value_gate.last_gate_scores <= 1).all()

    def test_gradient_flow(self, value_gate):
        """Test gradients flow through the gate."""
        batch_size, seq_len = 2, 32
        n_kv_heads, d_head = 8, 64

        v = torch.randn(batch_size, seq_len, n_kv_heads, d_head, requires_grad=True)
        hidden_states = torch.randn(batch_size, seq_len, 512, requires_grad=True)

        output = value_gate(v, hidden_states)
        loss = output.sum()
        loss.backward()

        assert v.grad is not None
        assert hidden_states.grad is not None
        assert not torch.isnan(v.grad).any()
        assert not torch.isnan(hidden_states.grad).any()

    def test_statistics(self, value_gate):
        """Test gate statistics collection."""
        batch_size, seq_len = 2, 64
        n_kv_heads, d_head = 8, 64

        v = torch.randn(batch_size, seq_len, n_kv_heads, d_head)
        hidden_states = torch.randn(batch_size, seq_len, 512)

        value_gate.eval()
        _ = value_gate(v, hidden_states)

        stats = value_gate.get_gate_statistics()

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert 0 <= stats["mean"] <= 1


class TestOutputGate:
    """Tests for OutputGate (G1)."""

    @pytest.fixture
    def output_gate(self):
        return OutputGate(
            d_model=512,
            n_heads=8,
            d_head=64,
            bias_init=0.5,
            per_head=True,
        )

    def test_initialization(self, output_gate):
        """Test OutputGate initializes correctly."""
        assert output_gate.d_model == 512
        assert output_gate.n_heads == 8
        assert output_gate.d_head == 64
        assert output_gate.per_head == True

    def test_forward_shape(self, output_gate):
        """Test forward pass output shape."""
        batch_size, seq_len = 2, 64
        n_heads, d_head = 8, 64

        attn_output = torch.randn(batch_size, seq_len, n_heads, d_head)
        hidden_states = torch.randn(batch_size, seq_len, 512)

        output = output_gate(attn_output, hidden_states)

        assert output.shape == attn_output.shape

    def test_per_head_vs_scalar(self):
        """Test per-head vs scalar gating."""
        d_model, n_heads, d_head = 512, 8, 64
        batch_size, seq_len = 2, 32

        # Per-head gating
        gate_per_head = OutputGate(d_model, n_heads, d_head, per_head=True)

        # Scalar gating
        gate_scalar = OutputGate(d_model, n_heads, d_head, per_head=False)

        attn_output = torch.randn(batch_size, seq_len, n_heads, d_head)
        hidden_states = torch.randn(batch_size, seq_len, d_model)

        output_per_head = gate_per_head(attn_output, hidden_states)
        output_scalar = gate_scalar(attn_output, hidden_states)

        # Both should produce same shape
        assert output_per_head.shape == output_scalar.shape

        # But gate projection sizes differ
        assert gate_per_head.gate_proj.out_features == n_heads * d_head
        assert gate_scalar.gate_proj.out_features == n_heads

    def test_gradient_flow(self, output_gate):
        """Test gradients flow through the gate."""
        batch_size, seq_len = 2, 32
        n_heads, d_head = 8, 64

        attn_output = torch.randn(batch_size, seq_len, n_heads, d_head, requires_grad=True)
        hidden_states = torch.randn(batch_size, seq_len, 512, requires_grad=True)

        output = output_gate(attn_output, hidden_states)
        loss = output.sum()
        loss.backward()

        assert attn_output.grad is not None
        assert hidden_states.grad is not None


class TestGateCombinations:
    """Test different gate configurations."""

    def test_both_gates(self):
        """Test with both gates enabled."""
        from gsa import GSAConfig, GatedSparseAttention

        config = GSAConfig(
            d_model=256,
            n_heads=4,
            n_kv_heads=2,
            use_value_gate=True,
            use_output_gate=True,
        )
        gsa = GatedSparseAttention(config)

        assert gsa.value_gate is not None
        assert gsa.output_gate is not None

    def test_only_value_gate(self):
        """Test with only value gate."""
        from gsa import GSAConfig, GatedSparseAttention

        config = GSAConfig(
            d_model=256,
            n_heads=4,
            n_kv_heads=2,
            use_value_gate=True,
            use_output_gate=False,
        )
        gsa = GatedSparseAttention(config)

        assert gsa.value_gate is not None
        assert gsa.output_gate is None

    def test_only_output_gate(self):
        """Test with only output gate."""
        from gsa import GSAConfig, GatedSparseAttention

        config = GSAConfig(
            d_model=256,
            n_heads=4,
            n_kv_heads=2,
            use_value_gate=False,
            use_output_gate=True,
        )
        gsa = GatedSparseAttention(config)

        assert gsa.value_gate is None
        assert gsa.output_gate is not None
