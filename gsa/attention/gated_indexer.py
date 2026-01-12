"""
Gated Lightning Indexer for efficient sparse token selection.

This module implements the gated indexer that computes importance scores
for each query-key pair to enable sparse attention. The key innovation
is using sigmoid gating instead of ReLU to produce bounded scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class GatedLightningIndexer(nn.Module):
    """
    Gated Lightning Indexer for efficient sparse token selection.

    Uses sigmoid gating instead of ReLU for:
    1. Bounded scores: sigmoid outputs are in [0, 1]
    2. Better gradient flow: sigmoid has non-zero gradients everywhere
    3. Query-dependent importance weighting

    Mathematical formulation:
        I_{t,s} = sum_{j=1}^{H^I} sigma(h_t W_j^{Iw}) * sigma(q_{t,j}^I * k_s^I + b_j^I)

    where:
        H^I: number of indexer heads
        h_t: hidden state at position t
        q_{t,j}^I: indexer query for head j
        k_s^I: indexer key at position s
        W_j^{Iw}: learnable weights for importance
        b_j^I: learnable bias
    """

    def __init__(
        self,
        d_model: int,
        d_indexer: int = 64,
        n_indexer_heads: int = 4,
        activation: str = "sigmoid",
        use_causal_mask: bool = True,
    ):
        """
        Initialize GatedLightningIndexer.

        Args:
            d_model: Input dimension
            d_indexer: Dimension of indexer queries/keys
            n_indexer_heads: Number of indexer heads
            activation: Activation for gating ("sigmoid" or "relu")
            use_causal_mask: Whether to apply causal masking
        """
        super().__init__()

        self.d_model = d_model
        self.d_indexer = d_indexer
        self.n_indexer_heads = n_indexer_heads
        self.activation = activation
        self.use_causal_mask = use_causal_mask

        # Query projection for indexer (per-head)
        self.q_proj = nn.Linear(d_model, n_indexer_heads * d_indexer, bias=False)

        # Key projection for indexer (shared across heads for efficiency)
        self.k_proj = nn.Linear(d_model, d_indexer, bias=False)

        # Query-dependent importance weights
        self.weight_proj = nn.Linear(d_model, n_indexer_heads, bias=True)

        # Learnable bias for adaptive thresholding
        self.bias = nn.Parameter(torch.zeros(n_indexer_heads))

        # Scale factor for dot product
        self.scale = 1.0 / math.sqrt(d_indexer)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with appropriate scales."""
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.weight_proj.weight, gain=0.1)
        nn.init.zeros_(self.weight_proj.bias)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        kv_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute gated indexer scores.

        Args:
            hidden_states: Query hidden states [batch, seq_q, d_model]
            positions: Position indices [batch, seq_q] (unused currently)
            kv_hidden_states: Key-Value hidden states [batch, seq_kv, d_model]
                             If None, uses hidden_states (self-attention)

        Returns:
            scores: [batch, seq_q, seq_kv] indexer scores for token selection
        """
        batch_size, seq_q, _ = hidden_states.shape

        # Use same hidden states for K if not provided (self-attention)
        if kv_hidden_states is None:
            kv_hidden_states = hidden_states
        seq_kv = kv_hidden_states.shape[1]

        # Project to indexer space
        q_idx = self.q_proj(hidden_states)  # [batch, seq_q, n_heads * d_idx]
        k_idx = self.k_proj(kv_hidden_states)  # [batch, seq_kv, d_idx]

        # Reshape queries
        q_idx = q_idx.view(batch_size, seq_q, self.n_indexer_heads, self.d_indexer)

        # Compute query-dependent importance weights
        weights = torch.sigmoid(self.weight_proj(hidden_states))  # [batch, seq_q, n_heads]

        # Compute raw scores for each indexer head
        # q_idx: [batch, seq_q, n_heads, d_idx]
        # k_idx: [batch, seq_kv, d_idx]
        raw_scores = torch.einsum('bqhd,bkd->bhqk', q_idx, k_idx) * self.scale

        # Apply activation with learnable bias
        # bias: [n_heads] -> [1, n_heads, 1, 1]
        bias = self.bias.view(1, -1, 1, 1)

        if self.activation == "sigmoid":
            gated_scores = torch.sigmoid(raw_scores + bias)
        elif self.activation == "relu":
            gated_scores = F.relu(raw_scores + bias)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Weight by query-dependent importance
        # weights: [batch, seq_q, n_heads] -> [batch, n_heads, seq_q, 1]
        weights_expanded = weights.permute(0, 2, 1).unsqueeze(-1)
        weighted_scores = gated_scores * weights_expanded

        # Sum across indexer heads
        final_scores = weighted_scores.sum(dim=1)  # [batch, seq_q, seq_kv]

        # Apply causal mask if needed
        if self.use_causal_mask and seq_q == seq_kv:
            causal_mask = torch.triu(
                torch.ones(seq_q, seq_kv, device=hidden_states.device, dtype=torch.bool),
                diagonal=1
            )
            final_scores = final_scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

        return final_scores

    def get_indexer_statistics(
        self,
        hidden_states: torch.Tensor,
    ) -> dict:
        """Get statistics about indexer behavior for analysis."""
        with torch.no_grad():
            scores = self.forward(hidden_states)

            # Compute statistics
            valid_scores = scores[scores != float('-inf')]

            return {
                "mean": valid_scores.mean().item(),
                "std": valid_scores.std().item(),
                "min": valid_scores.min().item(),
                "max": valid_scores.max().item(),
                "sparsity": (valid_scores < 0.1).float().mean().item(),
                "head_weights": torch.sigmoid(self.weight_proj.bias).tolist(),
            }
