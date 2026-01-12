"""
Rotary Position Embeddings (RoPE) implementation with support for various scaling methods.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Any, Tuple


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) with support for:
    - Standard RoPE
    - YaRN (Yet another RoPE extensioN) for context extension
    - NTK-aware interpolation
    - Dynamic NTK scaling
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: float = 10000.0,
        scaling_config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_config = scaling_config or {}

        # Compute inverse frequencies
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cos/sin cache
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=device,
            dtype=torch.float32,
        )

    def _compute_inv_freq(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Compute inverse frequencies with optional scaling."""
        scaling_type = self.scaling_config.get("type", None)

        if scaling_type == "yarn":
            return self._yarn_inv_freq(device)
        elif scaling_type == "ntk":
            return self._ntk_inv_freq(device)
        elif scaling_type == "dynamic_ntk":
            # Dynamic NTK is computed at runtime
            return self._standard_inv_freq(device)
        else:
            return self._standard_inv_freq(device)

    def _standard_inv_freq(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Standard RoPE inverse frequencies."""
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
        )
        return inv_freq

    def _ntk_inv_freq(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """NTK-aware interpolation inverse frequencies."""
        scaling_factor = self.scaling_config.get("factor", 1.0)
        base = self.base * scaling_factor ** (self.dim / (self.dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        return inv_freq

    def _yarn_inv_freq(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """YaRN (Yet another RoPE extensioN) inverse frequencies."""
        scaling_factor = self.scaling_config.get("factor", 1.0)
        original_max_position = self.scaling_config.get(
            "original_max_position_embeddings", 4096
        )
        beta_fast = self.scaling_config.get("beta_fast", 32)
        beta_slow = self.scaling_config.get("beta_slow", 1)
        mscale = self.scaling_config.get("mscale", 1.0)

        pos_freqs = self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low = max(int(math.floor(self.dim * math.log(original_max_position / (beta_fast * 2 * math.pi)) / (2 * math.log(self.base)))), 0)
        high = min(int(math.ceil(self.dim * math.log(original_max_position / (beta_slow * 2 * math.pi)) / (2 * math.log(self.base)))), self.dim // 2 - 1)

        # Smooth interpolation
        inv_freq = inv_freq_interpolation.clone()
        inv_freq[:low] = inv_freq_extrapolation[:low]

        if high > low:
            smooth = (torch.arange(low, high + 1, device=device).float() - low) / (high - low)
            inv_freq[low:high + 1] = (
                (1 - smooth) * inv_freq_extrapolation[low:high + 1]
                + smooth * inv_freq_interpolation[low:high + 1]
            )

        return inv_freq

    def _set_cos_sin_cache(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Precompute cos and sin cache."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin for given positions.

        Args:
            positions: [batch, seq_len] position indices
            seq_len: Optional sequence length for cache extension

        Returns:
            cos: [batch, seq_len, dim]
            sin: [batch, seq_len, dim]
        """
        if seq_len is None:
            seq_len = positions.max().item() + 1

        # Extend cache if necessary
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(
                seq_len,
                device=positions.device,
                dtype=self.cos_cached.dtype,
            )

        # Gather cos and sin for the given positions
        cos = self.cos_cached[positions]  # [batch, seq_len, dim]
        sin = self.sin_cached[positions]  # [batch, seq_len, dim]

        return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: [batch, seq_len, n_heads, d_head]
        k: [batch, seq_len, n_kv_heads, d_head]
        cos: [batch, seq_len, d_head]
        sin: [batch, seq_len, d_head]

    Returns:
        q_rotated: [batch, seq_len, n_heads, d_head]
        k_rotated: [batch, seq_len, n_kv_heads, d_head]
    """
    # Expand cos/sin for head dimension
    cos = cos.unsqueeze(2)  # [batch, seq_len, 1, d_head]
    sin = sin.unsqueeze(2)  # [batch, seq_len, 1, d_head]

    q_rotated = _rotate_half(q, cos, sin)
    k_rotated = _rotate_half(k, cos, sin)

    return q_rotated, k_rotated


def _rotate_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin
