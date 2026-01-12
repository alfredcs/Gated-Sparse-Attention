"""
Gated Sparse Attention (GSA) - Main implementation.

This module implements the core GSA mechanism that combines:
1. Gated Lightning Indexer for efficient token selection
2. Adaptive Top-K sparse attention
3. Dual gating mechanism (G1 output gate + G2 value gate)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from .gated_indexer import GatedLightningIndexer
from .adaptive_topk import AdaptiveTopKSelector
from .value_gate import ValueGate
from .output_gate import OutputGate
from .rope import RotaryEmbedding, apply_rotary_pos_emb


class GatedSparseAttention(nn.Module):
    """
    Gated Sparse Attention (GSA) Module.

    Combines sparse token selection with dual gating for efficient
    and stable long-context attention.

    Key innovations:
    1. Gated Lightning Indexer: Uses sigmoid activation for bounded scores
    2. Adaptive Sparsity: Dynamically adjusts k based on attention distribution
    3. Dual Gating: G1 (output gate) + G2 (value gate) for stability and sink elimination

    Computational complexity: O(L * k * d) instead of O(L^2 * d)
    where L is sequence length, k is selected tokens, d is dimension
    """

    def __init__(self, config: 'GSAConfig'):
        """
        Initialize GatedSparseAttention.

        Args:
            config: GSAConfig with all hyperparameters
        """
        super().__init__()
        self.config = config

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_head
        self.n_rep = self.n_heads // self.n_kv_heads

        self.scale = 1.0 / math.sqrt(self.d_head)

        # QKV Projections
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.d_head, self.d_model, bias=False)

        # Gated Lightning Indexer
        self.indexer = GatedLightningIndexer(
            d_model=self.d_model,
            d_indexer=config.d_indexer,
            n_indexer_heads=config.n_indexer_heads,
            activation=config.indexer_activation,
        )

        # Adaptive Top-K Selector
        self.topk_selector = AdaptiveTopKSelector(
            k_base=config.k_base,
            k_min=config.k_min,
            k_max=config.k_max,
            use_adaptive=config.use_adaptive_k,
            temperature=config.adaptive_k_temperature,
        )

        # Gating modules
        if config.use_value_gate:
            self.value_gate = ValueGate(
                d_model=self.d_model,
                n_kv_heads=self.n_kv_heads,
                d_head=self.d_head,
                bias_init=config.gate_bias_init,
            )
        else:
            self.value_gate = None

        if config.use_output_gate:
            self.output_gate = OutputGate(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_head=self.d_head,
                bias_init=config.gate_bias_init,
            )
        else:
            self.output_gate = None

        # Position embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=self.d_head,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_base,
            scaling_config=config.rope_scaling,
        )

        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)

        # For training stability during warm-up
        self.dense_residual_alpha = config.dense_residual_alpha

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with appropriate scales."""
        for module in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(module.weight, gain=1.0 / math.sqrt(2))
        nn.init.xavier_uniform_(self.o_proj.weight, gain=1.0 / math.sqrt(2 * self.config.n_layers))

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repeat KV heads to match query heads for GQA.

        Args:
            x: [batch, seq_len, n_kv_heads, d_head]

        Returns:
            [batch, seq_len, n_heads, d_head]
        """
        batch, seq_len, n_kv_heads, d_head = x.shape
        if self.n_rep == 1:
            return x
        x = x.unsqueeze(3).expand(batch, seq_len, n_kv_heads, self.n_rep, d_head)
        return x.reshape(batch, seq_len, self.n_heads, d_head)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple], Optional[torch.Tensor]]:
        """
        Forward pass for Gated Sparse Attention.

        Args:
            hidden_states: [batch, seq_len, d_model]
            positions: [batch, seq_len] position indices
            attention_mask: Optional attention mask
            past_key_value: Optional KV cache tuple (k, v)
            use_cache: Whether to return updated cache
            output_attentions: Whether to return attention weights

        Returns:
            output: [batch, seq_len, d_model]
            past_key_value: Updated cache (if use_cache)
            attentions: Attention weights (if output_attentions)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Generate positions if not provided
        if positions is None:
            past_len = past_key_value[0].shape[1] if past_key_value else 0
            positions = torch.arange(
                past_len, past_len + seq_len,
                device=hidden_states.device
            ).unsqueeze(0).expand(batch_size, -1)

        # === Step 1: QKV Projections ===
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.n_heads, self.d_head)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.n_kv_heads, self.d_head)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.n_kv_heads, self.d_head)

        # === Step 2: Apply Value Gate (G2) ===
        if self.value_gate is not None:
            v = self.value_gate(v, hidden_states)

        # === Step 3: Apply RoPE ===
        cos, sin = self.rotary_emb(positions)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # === Step 4: Handle KV Cache ===
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)

        present_key_value = (k, v) if use_cache else None
        kv_seq_len = k.shape[1]

        # === Step 5: Compute Gated Indexer Scores ===
        # For cached inference, we only compute indexer for new tokens
        indexer_scores = self.indexer(
            hidden_states,
            positions,
            kv_hidden_states=None,  # Will need full KV hidden states for proper indexing
        )

        # Extend indexer scores for KV cache case
        if past_key_value is not None and kv_seq_len > seq_len:
            # Need to extend scores to cover all KV positions
            # For simplicity, we pad with zeros (to be improved with proper caching)
            padding = torch.zeros(
                batch_size, seq_len, kv_seq_len - seq_len,
                device=indexer_scores.device, dtype=indexer_scores.dtype
            )
            # Actually, we should compute scores against all KV positions
            # This is a simplified version - full implementation would cache indexer keys

        # === Step 6: Adaptive Top-k Selection ===
        selected_indices, selection_mask = self.topk_selector(
            indexer_scores, seq_len, indexer_scores.shape[-1]
        )

        # === Step 7: Sparse Attention ===
        attn_output, attn_weights = self._sparse_attention(
            q, k, v, selected_indices, selection_mask
        )

        # === Step 8: Apply Output Gate (G1) ===
        if self.output_gate is not None:
            attn_output = self.output_gate(attn_output, hidden_states)

        # === Step 9: Output Projection ===
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)

        return output, present_key_value, attn_weights if output_attentions else None

    def _sparse_attention(
        self,
        q: torch.Tensor,         # [batch, seq_q, n_heads, d_head]
        k: torch.Tensor,         # [batch, seq_kv, n_kv_heads, d_head]
        v: torch.Tensor,         # [batch, seq_kv, n_kv_heads, d_head]
        indices: torch.Tensor,   # [batch, seq_q, k_selected]
        mask: torch.Tensor,      # [batch, seq_q, k_selected]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute sparse attention using selected indices.

        Args:
            q: Query tensor [batch, seq_q, n_heads, d_head]
            k: Key tensor [batch, seq_kv, n_kv_heads, d_head]
            v: Value tensor [batch, seq_kv, n_kv_heads, d_head]
            indices: Selected token indices [batch, seq_q, k_selected]
            mask: Valid selection mask [batch, seq_q, k_selected]

        Returns:
            output: [batch, seq_q, n_heads, d_head]
            attn_weights: [batch, seq_q, n_heads, k_selected] (optional)
        """
        batch_size, seq_q, n_heads, d_head = q.shape
        k_selected = indices.shape[-1]

        # Repeat KV for GQA
        k = self._repeat_kv(k)  # [batch, seq_kv, n_heads, d_head]
        v = self._repeat_kv(v)  # [batch, seq_kv, n_heads, d_head]

        # Gather selected K and V
        # indices: [batch, seq_q, k_selected]
        k_gathered = self._gather_along_seq(k, indices)  # [batch, seq_q, k_selected, n_heads, d_head]
        v_gathered = self._gather_along_seq(v, indices)  # [batch, seq_q, k_selected, n_heads, d_head]

        # Compute attention scores
        # q: [batch, seq_q, n_heads, d_head] -> [batch, seq_q, n_heads, 1, d_head]
        # k_gathered: [batch, seq_q, k_selected, n_heads, d_head] -> [batch, seq_q, n_heads, k_selected, d_head]
        k_gathered = k_gathered.permute(0, 1, 3, 2, 4)  # [batch, seq_q, n_heads, k_selected, d_head]
        v_gathered = v_gathered.permute(0, 1, 3, 2, 4)  # [batch, seq_q, n_heads, k_selected, d_head]

        # Attention scores: [batch, seq_q, n_heads, k_selected]
        scores = torch.einsum('bqhd,bqhkd->bqhk', q, k_gathered) * self.scale

        # Apply mask
        # mask: [batch, seq_q, k_selected] -> [batch, seq_q, 1, k_selected]
        mask_expanded = mask.unsqueeze(2)
        scores = scores.masked_fill(~mask_expanded, float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_weights.masked_fill(~mask_expanded, 0.0)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum: [batch, seq_q, n_heads, d_head]
        output = torch.einsum('bqhk,bqhkd->bqhd', attn_weights, v_gathered)

        return output, attn_weights

    def _gather_along_seq(
        self,
        x: torch.Tensor,       # [batch, seq_kv, n_heads, d_head]
        indices: torch.Tensor,  # [batch, seq_q, k_selected]
    ) -> torch.Tensor:
        """
        Gather tokens along sequence dimension.

        Args:
            x: Input tensor [batch, seq_kv, n_heads, d_head]
            indices: Indices to gather [batch, seq_q, k_selected]

        Returns:
            Gathered tensor [batch, seq_q, k_selected, n_heads, d_head]
        """
        batch, seq_kv, n_heads, d_head = x.shape
        _, seq_q, k_selected = indices.shape

        # Expand indices for gathering
        # indices: [batch, seq_q, k_selected] -> [batch, seq_q, k_selected, n_heads, d_head]
        indices_expanded = indices.unsqueeze(-1).unsqueeze(-1).expand(
            batch, seq_q, k_selected, n_heads, d_head
        )

        # Expand x for gathering
        # x: [batch, seq_kv, n_heads, d_head] -> [batch, 1, seq_kv, n_heads, d_head]
        x_expanded = x.unsqueeze(1).expand(batch, seq_q, seq_kv, n_heads, d_head)

        # Gather
        gathered = torch.gather(x_expanded, 2, indices_expanded)

        return gathered
