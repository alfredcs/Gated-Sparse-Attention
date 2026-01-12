"""
Value Gate (G2) implementation for Gated Sparse Attention.

The value gate is applied after the value projection to control
information flow before attention computation. This helps eliminate
attention sinks by allowing the model to down-weight certain value
representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ValueGate(nn.Module):
    """
    G2 Position: Gate applied after value projection.

    This gate modulates the value vectors before they are used in
    attention computation. By learning to gate values based on the
    input hidden states, the model can:
    1. Reduce reliance on positional tokens (eliminating attention sinks)
    2. Create sparser effective value representations
    3. Improve gradient flow during training

    Mathematical formulation:
        v_gated = v * sigmoid(W_g * h + b_g)

    where:
        v: value projection output [batch, seq, n_kv_heads, d_head]
        h: input hidden states [batch, seq, d_model]
        W_g: gate projection weights
        b_g: gate bias (initialized for moderate gating)
    """

    def __init__(
        self,
        d_model: int,
        n_kv_heads: int,
        d_head: int,
        bias_init: float = 0.5,
        activation: str = "sigmoid",
    ):
        """
        Initialize ValueGate.

        Args:
            d_model: Input dimension (hidden state dimension)
            n_kv_heads: Number of key-value heads
            d_head: Dimension per head
            bias_init: Initial bias value (controls initial gate openness)
            activation: Gate activation function ("sigmoid", "tanh", "silu")
        """
        super().__init__()

        self.d_model = d_model
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.activation = activation

        # Gate projection: hidden_states -> gate_scores
        self.gate_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=True)

        # Initialize for moderate gating
        # Positive bias means gates start more open (sigmoid(0.5) ≈ 0.62)
        self._init_weights(bias_init)

        # Store last gate scores for analysis
        self.last_gate_scores: Optional[torch.Tensor] = None

    def _init_weights(self, bias_init: float):
        """Initialize weights and bias."""
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.1)
        nn.init.constant_(self.gate_proj.bias, bias_init)

    def forward(
        self,
        v: torch.Tensor,              # [batch, seq, n_kv_heads, d_head]
        hidden_states: torch.Tensor,  # [batch, seq, d_model]
    ) -> torch.Tensor:
        """
        Apply value gating.

        Args:
            v: Value projection output [batch, seq, n_kv_heads, d_head]
            hidden_states: Input hidden states [batch, seq, d_model]

        Returns:
            Gated values [batch, seq, n_kv_heads, d_head]
        """
        batch_size, seq_len, n_kv_heads, d_head = v.shape

        # Compute gate scores
        gate_logits = self.gate_proj(hidden_states)
        gate_logits = gate_logits.view(batch_size, seq_len, n_kv_heads, d_head)

        # Apply activation
        if self.activation == "sigmoid":
            gate = torch.sigmoid(gate_logits)
        elif self.activation == "tanh":
            gate = (torch.tanh(gate_logits) + 1) / 2  # Scale to [0, 1]
        elif self.activation == "silu":
            gate = F.silu(gate_logits)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Store for analysis (detached to avoid memory issues)
        if not self.training:
            self.last_gate_scores = gate.detach()

        # Apply gate
        return v * gate

    def get_gate_statistics(self) -> dict:
        """Get statistics about gate activations (for analysis)."""
        if self.last_gate_scores is None:
            return {}

        scores = self.last_gate_scores
        return {
            "mean": scores.mean().item(),
            "std": scores.std().item(),
            "min": scores.min().item(),
            "max": scores.max().item(),
            "sparsity": (scores < 0.1).float().mean().item(),  # Fraction near-zero
        }
