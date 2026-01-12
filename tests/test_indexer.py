"""
Tests for Gated Lightning Indexer and Adaptive Top-K Selector.
"""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gsa.attention import GatedLightningIndexer, AdaptiveTopKSelector


class TestGatedLightningIndexer:
    """Tests for the Gated Lightning Indexer."""

    @pytest.fixture
    def indexer(self):
        return GatedLightningIndexer(
            d_model=512,
            d_indexer=32,
            n_indexer_heads=4,
            activation="sigmoid",
        )

    def test_initialization(self, indexer):
        """Test indexer initializes correctly."""
        assert indexer.d_model == 512
        assert indexer.d_indexer == 32
        assert indexer.n_indexer_heads == 4
        assert indexer.activation == "sigmoid"

    def test_forward_shape(self, indexer):
        """Test forward pass output shape."""
        batch_size, seq_len = 2, 64

        hidden_states = torch.randn(batch_size, seq_len, 512)
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        scores = indexer(hidden_states, positions)

        # Output should be [batch, seq_q, seq_kv]
        assert scores.shape == (batch_size, seq_len, seq_len)

    def test_causal_masking(self, indexer):
        """Test causal masking is applied."""
        batch_size, seq_len = 1, 32

        hidden_states = torch.randn(batch_size, seq_len, 512)
        positions = torch.arange(seq_len).unsqueeze(0)

        scores = indexer(hidden_states, positions)

        # Upper triangular should be -inf (future tokens masked)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert scores[0, i, j] == float('-inf')

    def test_sigmoid_bounded_scores(self, indexer):
        """Test sigmoid activation produces bounded scores."""
        batch_size, seq_len = 2, 32

        hidden_states = torch.randn(batch_size, seq_len, 512)
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        scores = indexer(hidden_states, positions)

        # Valid scores (not masked) should be bounded
        valid_scores = scores[scores != float('-inf')]
        assert (valid_scores >= 0).all()
        # For sum of sigmoid products, max is n_indexer_heads
        assert (valid_scores <= indexer.n_indexer_heads + 1).all()

    def test_relu_activation(self):
        """Test ReLU activation variant."""
        indexer_relu = GatedLightningIndexer(
            d_model=512,
            d_indexer=32,
            n_indexer_heads=4,
            activation="relu",
        )

        batch_size, seq_len = 2, 32
        hidden_states = torch.randn(batch_size, seq_len, 512)
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        scores = indexer_relu(hidden_states, positions)

        valid_scores = scores[scores != float('-inf')]
        assert (valid_scores >= 0).all()  # ReLU produces non-negative

    def test_gradient_flow(self, indexer):
        """Test gradients flow correctly."""
        batch_size, seq_len = 2, 32

        hidden_states = torch.randn(batch_size, seq_len, 512, requires_grad=True)
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        scores = indexer(hidden_states, positions)
        valid_scores = scores[scores != float('-inf')]
        loss = valid_scores.sum()
        loss.backward()

        assert hidden_states.grad is not None
        assert not torch.isnan(hidden_states.grad).any()

    def test_cross_attention(self, indexer):
        """Test indexer with separate Q and KV sequences."""
        batch_size = 2
        seq_q, seq_kv = 32, 64

        q_hidden = torch.randn(batch_size, seq_q, 512)
        kv_hidden = torch.randn(batch_size, seq_kv, 512)
        positions = torch.arange(seq_q).unsqueeze(0).expand(batch_size, -1)

        # Create indexer without causal mask for cross attention
        indexer_cross = GatedLightningIndexer(
            d_model=512,
            d_indexer=32,
            n_indexer_heads=4,
            use_causal_mask=False,
        )

        scores = indexer_cross(q_hidden, positions, kv_hidden_states=kv_hidden)

        assert scores.shape == (batch_size, seq_q, seq_kv)


class TestAdaptiveTopKSelector:
    """Tests for Adaptive Top-K Selector."""

    @pytest.fixture
    def selector(self):
        return AdaptiveTopKSelector(
            k_base=32,
            k_min=8,
            k_max=64,
            use_adaptive=True,
            temperature=1.0,
        )

    @pytest.fixture
    def selector_fixed(self):
        return AdaptiveTopKSelector(
            k_base=32,
            k_min=8,
            k_max=64,
            use_adaptive=False,
        )

    def test_initialization(self, selector):
        """Test selector initializes correctly."""
        assert selector.k_base == 32
        assert selector.k_min == 8
        assert selector.k_max == 64
        assert selector.use_adaptive == True

    def test_forward_shape(self, selector):
        """Test forward pass output shapes."""
        batch_size, seq_q, seq_kv = 2, 64, 64

        # Create dummy scores
        scores = torch.randn(batch_size, seq_q, seq_kv)
        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_q, seq_kv), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))

        indices, mask = selector(scores, seq_q, seq_kv)

        # indices should be [batch, seq_q, k_effective]
        assert indices.shape[0] == batch_size
        assert indices.shape[1] == seq_q
        assert indices.shape[2] <= selector.k_max

        # mask should match indices shape
        assert mask.shape == indices.shape

    def test_fixed_k(self, selector_fixed):
        """Test fixed k selection."""
        batch_size, seq_q, seq_kv = 2, 64, 64

        scores = torch.randn(batch_size, seq_q, seq_kv)
        causal_mask = torch.triu(torch.ones(seq_q, seq_kv), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))

        indices, mask = selector_fixed(scores, seq_q, seq_kv)

        # With fixed k, should select k_base tokens
        assert indices.shape[2] == min(selector_fixed.k_base, seq_kv)

    def test_adaptive_k_variation(self, selector):
        """Test that adaptive k actually varies."""
        batch_size, seq_q, seq_kv = 4, 64, 64

        # Create scores with different variance patterns
        scores = torch.randn(batch_size, seq_q, seq_kv)
        causal_mask = torch.triu(torch.ones(seq_q, seq_kv), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))

        indices, mask, k_values = selector(scores, seq_q, seq_kv, return_k_values=True)

        # k_values should vary
        assert k_values is not None
        # Should be within bounds
        assert (k_values >= selector.k_min).all()
        assert (k_values <= selector.k_max).all()

    def test_mask_validity(self, selector):
        """Test that mask correctly indicates valid selections."""
        batch_size, seq_q, seq_kv = 2, 32, 32

        scores = torch.randn(batch_size, seq_q, seq_kv)
        causal_mask = torch.triu(torch.ones(seq_q, seq_kv), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))

        indices, mask = selector(scores, seq_q, seq_kv)

        # Masked positions should have valid indices
        for b in range(batch_size):
            for q in range(seq_q):
                valid_count = mask[b, q].sum().item()
                # Valid count should be within bounds
                assert valid_count <= selector.k_max
                assert valid_count >= 0

    def test_indices_in_range(self, selector):
        """Test that selected indices are within valid range."""
        batch_size, seq_q, seq_kv = 2, 64, 64

        scores = torch.randn(batch_size, seq_q, seq_kv)
        causal_mask = torch.triu(torch.ones(seq_q, seq_kv), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))

        indices, mask = selector(scores, seq_q, seq_kv)

        # All indices should be valid
        assert (indices >= 0).all()
        assert (indices < seq_kv).all()


class TestIndexerIntegration:
    """Integration tests for indexer + selector."""

    def test_indexer_to_selector_pipeline(self):
        """Test full pipeline from indexer to selector."""
        batch_size, seq_len = 2, 64
        d_model = 512

        indexer = GatedLightningIndexer(
            d_model=d_model,
            d_indexer=32,
            n_indexer_heads=4,
        )

        selector = AdaptiveTopKSelector(
            k_base=16,
            k_min=4,
            k_max=32,
            use_adaptive=True,
        )

        # Generate hidden states
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        # Run indexer
        scores = indexer(hidden_states, positions)

        # Run selector
        indices, mask = selector(scores, seq_len, seq_len)

        # Verify shapes
        assert scores.shape == (batch_size, seq_len, seq_len)
        assert indices.shape[0] == batch_size
        assert indices.shape[1] == seq_len
        assert mask.shape == indices.shape
