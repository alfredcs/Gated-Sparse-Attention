"""
Adaptive Top-K Token Selector for Gated Sparse Attention.

This module implements adaptive selection of tokens based on indexer scores,
with the number of selected tokens varying based on score variance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AdaptiveTopKSelector(nn.Module):
    """
    Adaptive Top-K Token Selector.

    Selects the most relevant tokens for each query based on indexer scores.
    The number of selected tokens can be:
    1. Fixed: Always select k_base tokens
    2. Adaptive: Vary k based on score variance (more uniform scores -> more tokens)

    Mathematical formulation for adaptive k:
        k_t = f(Var(I_{t,:}))

    where k_t is clamped to [k_min, k_max]
    """

    def __init__(
        self,
        k_base: int = 2048,
        k_min: int = 256,
        k_max: int = 4096,
        use_adaptive: bool = True,
        temperature: float = 1.0,
        adaptive_method: str = "variance",
    ):
        """
        Initialize AdaptiveTopKSelector.

        Args:
            k_base: Base number of tokens to select
            k_min: Minimum number of tokens
            k_max: Maximum number of tokens
            use_adaptive: Whether to use adaptive k selection
            temperature: Temperature for adaptive scaling
            adaptive_method: Method for computing adaptive k
                           ("variance", "entropy", "learned")
        """
        super().__init__()

        self.k_base = k_base
        self.k_min = k_min
        self.k_max = k_max
        self.use_adaptive = use_adaptive
        self.temperature = temperature
        self.adaptive_method = adaptive_method

        # For learned adaptive method
        if adaptive_method == "learned":
            self.k_predictor = nn.Sequential(
                nn.Linear(3, 32),  # Input: [mean, std, max]
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

    def _compute_adaptive_k(
        self,
        scores: torch.Tensor,  # [batch, seq_q, seq_kv]
    ) -> torch.Tensor:
        """
        Compute adaptive k values based on score distribution.

        Args:
            scores: Indexer scores [batch, seq_q, seq_kv]

        Returns:
            k_values: Per-query k values [batch, seq_q]
        """
        batch_size, seq_q, seq_kv = scores.shape

        # Mask out invalid scores (causal mask results in -inf)
        valid_mask = scores != float('-inf')
        masked_scores = scores.masked_fill(~valid_mask, 0)

        # Count valid positions per query
        valid_counts = valid_mask.sum(dim=-1).float()  # [batch, seq_q]

        if self.adaptive_method == "variance":
            # Higher variance -> more focused attention -> can use fewer tokens
            # Lower variance -> more spread attention -> need more tokens
            score_var = masked_scores.var(dim=-1)  # [batch, seq_q]

            # Normalize variance
            var_normalized = score_var / (score_var.mean() + 1e-8)

            # Inverse relationship: high variance -> low k
            k_scale = 1.0 / (1.0 + var_normalized * self.temperature)

            # Scale k_base
            k_adaptive = self.k_base * (0.5 + k_scale)  # Range: [0.5*k_base, 1.5*k_base]

        elif self.adaptive_method == "entropy":
            # Compute softmax entropy
            probs = F.softmax(masked_scores.masked_fill(~valid_mask, float('-inf')), dim=-1)
            probs = probs.masked_fill(~valid_mask, 0)
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=-1)  # [batch, seq_q]

            # Max entropy is log(valid_counts)
            max_entropy = torch.log(valid_counts + 1)
            normalized_entropy = entropy / (max_entropy + 1e-8)

            # High entropy (uniform) -> need more tokens
            k_adaptive = self.k_base * (0.5 + normalized_entropy * self.temperature)

        elif self.adaptive_method == "learned":
            # Compute statistics for learned predictor
            score_mean = masked_scores.mean(dim=-1, keepdim=True)
            score_std = masked_scores.std(dim=-1, keepdim=True)
            score_max = masked_scores.max(dim=-1, keepdim=True).values
            stats = torch.cat([score_mean, score_std, score_max], dim=-1)  # [batch, seq_q, 3]

            # Predict k scale
            k_scale = self.k_predictor(stats).squeeze(-1)  # [batch, seq_q]
            k_adaptive = self.k_min + k_scale * (self.k_max - self.k_min)

        else:
            raise ValueError(f"Unknown adaptive method: {self.adaptive_method}")

        # Clamp to valid range and ensure <= valid positions
        k_values = torch.clamp(k_adaptive, self.k_min, self.k_max)
        k_values = torch.minimum(k_values, valid_counts)

        return k_values.long()

    def forward(
        self,
        scores: torch.Tensor,        # [batch, seq_q, seq_kv]
        seq_q: int,
        seq_kv: int,
        return_k_values: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k tokens for each query position.

        Args:
            scores: Indexer scores [batch, seq_q, seq_kv]
            seq_q: Query sequence length
            seq_kv: Key-Value sequence length
            return_k_values: Whether to return actual k values used

        Returns:
            indices: Selected token indices [batch, seq_q, k_effective]
            mask: Valid selection mask [batch, seq_q, k_effective]
            k_values: (optional) Actual k values [batch, seq_q]
        """
        batch_size = scores.shape[0]

        if self.use_adaptive:
            k_values = self._compute_adaptive_k(scores)
            # Use max k for tensor shape, mask invalid positions
            k_effective = min(self.k_max, seq_kv)
        else:
            k_values = torch.full(
                (batch_size, seq_q),
                min(self.k_base, seq_kv),
                device=scores.device,
                dtype=torch.long,
            )
            k_effective = min(self.k_base, seq_kv)

        # Get top-k indices
        # Handle -inf values by replacing with very negative number
        scores_for_topk = scores.masked_fill(scores == float('-inf'), -1e9)
        _, indices = torch.topk(scores_for_topk, k_effective, dim=-1)

        # Create mask for valid selections
        # indices shape: [batch, seq_q, k_effective]
        position_indices = torch.arange(k_effective, device=scores.device)
        position_indices = position_indices.view(1, 1, -1).expand(batch_size, seq_q, -1)

        # Mask positions beyond k_values for each query
        mask = position_indices < k_values.unsqueeze(-1)

        # Also mask based on original scores (positions that were -inf)
        gathered_scores = torch.gather(scores, -1, indices)
        mask = mask & (gathered_scores != float('-inf'))

        if return_k_values:
            return indices, mask, k_values
        return indices, mask

    def get_selection_statistics(
        self,
        scores: torch.Tensor,
    ) -> dict:
        """Get statistics about token selection for analysis."""
        with torch.no_grad():
            indices, mask, k_values = self.forward(
                scores, scores.shape[1], scores.shape[2], return_k_values=True
            )

            return {
                "k_mean": k_values.float().mean().item(),
                "k_std": k_values.float().std().item(),
                "k_min": k_values.min().item(),
                "k_max": k_values.max().item(),
                "coverage": mask.float().mean().item(),
            }
