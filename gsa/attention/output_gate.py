"""
Output Gate (G1) implementation for Gated Sparse Attention.

The output gate is applied after SDPA (Scaled Dot-Product Attention)
computation. This is the most effective position for gating as it
operates on the attention output directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class OutputGate(nn.Module):
    """
    G1 Position: Gate applied after SDPA output (most effective position).

    This gate modulates the attention output per-head, allowing the model to:
    1. Dynamically suppress or amplify attention outputs
    2. Eliminate attention sinks by reducing reliance on specific patterns
    3. Improve gradient stability through bounded activations

    Mathematical formulation:
        o_gated = o * sigmoid(W_g * h + b_g)

    where:
        o: attention output [batch, seq, n_heads, d_head]
        h: input hidden states [batch, seq, d_model]
        W_g: gate projection weights
        b_g: gate bias (initialized for moderate gating)

    Key insight from ablation studies:
        G1 (output gate) is more effective than G2 (value gate) for:
        - Eliminating attention sinks
        - Improving training stability
        - Better downstream task performance
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        bias_init: float = 0.5,
        activation: str = "sigmoid",
        per_head: bool = True,
    ):
        """
        Initialize OutputGate.

        Args:
            d_model: Input dimension (hidden state dimension)
            n_heads: Number of attention heads
            d_head: Dimension per head
            bias_init: Initial bias value (controls initial gate openness)
            activation: Gate activation function
            per_head: If True, compute per-head gates (recommended)
        """
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.activation = activation
        self.per_head = per_head

        if per_head:
            # Per-head gating (more expressive)
            self.gate_proj = nn.Linear(d_model, n_heads * d_head, bias=True)
        else:
            # Per-head scalar gating (more efficient)
            self.gate_proj = nn.Linear(d_model, n_heads, bias=True)

        # Initialize for moderate gating
        self._init_weights(bias_init)

        # Store last gate scores for analysis
        self.last_gate_scores: Optional[torch.Tensor] = None

    def _init_weights(self, bias_init: float):
        """Initialize weights and bias."""
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.1)
        nn.init.constant_(self.gate_proj.bias, bias_init)

    def forward(
        self,
        attn_output: torch.Tensor,    # [batch, seq, n_heads, d_head]
        hidden_states: torch.Tensor,  # [batch, seq, d_model]
    ) -> torch.Tensor:
        """
        Apply output gating.

        Args:
            attn_output: Attention output [batch, seq, n_heads, d_head]
            hidden_states: Input hidden states [batch, seq, d_model]

        Returns:
            Gated attention output [batch, seq, n_heads, d_head]
        """
        batch_size, seq_len, n_heads, d_head = attn_output.shape

        # Compute gate scores
        gate_logits = self.gate_proj(hidden_states)

        if self.per_head:
            gate_logits = gate_logits.view(batch_size, seq_len, n_heads, d_head)
        else:
            gate_logits = gate_logits.view(batch_size, seq_len, n_heads, 1)

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
        return attn_output * gate

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
            "sparsity": (scores < 0.1).float().mean().item(),
            # Per-head statistics
            "per_head_mean": scores.mean(dim=(0, 1, 3)).tolist() if scores.dim() == 4 else None,
        }
