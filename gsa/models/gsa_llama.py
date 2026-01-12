"""
GSA-enhanced LLaMA model implementation.

This module provides a complete LLaMA-style model with GSA attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
import math

from ..config import GSAConfig
from ..attention import GatedSparseAttention


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


class GSAMlp(nn.Module):
    """MLP block with SwiGLU activation."""

    def __init__(self, config: GSAConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_ffn = config.d_ffn

        self.gate_proj = nn.Linear(self.d_model, self.d_ffn, bias=False)
        self.up_proj = nn.Linear(self.d_model, self.d_ffn, bias=False)
        self.down_proj = nn.Linear(self.d_ffn, self.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GSADecoderLayer(nn.Module):
    """Single decoder layer with GSA attention and MLP."""

    def __init__(self, config: GSAConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-attention normalization
        self.input_layernorm = RMSNorm(config.d_model, eps=config.norm_eps)

        # Gated Sparse Attention
        self.self_attn = GatedSparseAttention(config)

        # Post-attention normalization
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.norm_eps)

        # MLP
        self.mlp = GSAMlp(config)

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
        Forward pass for decoder layer.

        Args:
            hidden_states: [batch, seq_len, d_model]
            positions: Position indices
            attention_mask: Attention mask
            past_key_value: KV cache
            use_cache: Whether to return updated cache
            output_attentions: Whether to return attention weights

        Returns:
            hidden_states: Updated hidden states
            present_key_value: Updated KV cache
            attentions: Attention weights (optional)
        """
        residual = hidden_states

        # Pre-norm + GSA
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, present_key_value, attn_weights = self.self_attn(
            hidden_states,
            positions=positions,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = residual + attn_output

        # Pre-norm + MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, present_key_value, attn_weights


@dataclass
class GSAModelOutput:
    """Output class for GSA models."""
    last_hidden_state: torch.Tensor
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


@dataclass
class GSACausalLMOutput:
    """Output class for GSA causal language models."""
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class GSALlamaModel(nn.Module):
    """GSA-enhanced LLaMA base model (without LM head)."""

    def __init__(self, config: GSAConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Decoder layers
        self.layers = nn.ModuleList([
            GSADecoderLayer(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # Gradient checkpointing
        self.gradient_checkpointing = False

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        std = self.config.initializer_range
        nn.init.normal_(self.embed_tokens.weight, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> GSAModelOutput:
        """
        Forward pass for GSA LLaMA model.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            positions: Position indices
            attention_mask: Attention mask
            past_key_values: KV cache for all layers
            use_cache: Whether to return updated cache
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states

        Returns:
            GSAModelOutput with last_hidden_state and optional outputs
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Generate positions if not provided
        if positions is None:
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            positions = torch.arange(
                past_len, past_len + seq_len,
                device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        # Storage for outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_cache = () if use_cache else None

        # Process through decoder layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values else None

            if self.gradient_checkpointing and self.training:
                hidden_states, present_key_value, attn_weights = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    positions,
                    attention_mask,
                    past_key_value,
                    use_cache,
                    output_attentions,
                    use_reentrant=False,
                )
            else:
                hidden_states, present_key_value, attn_weights = layer(
                    hidden_states,
                    positions=positions,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            if use_cache:
                next_cache = next_cache + (present_key_value,)

            if output_attentions:
                all_attentions = all_attentions + (attn_weights,)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return GSAModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class GSALlamaForCausalLM(nn.Module):
    """GSA-enhanced LLaMA for causal language modeling."""

    def __init__(self, config: GSAConfig):
        super().__init__()
        self.config = config

        # Base model
        self.model = GSALlamaModel(config)

        # LM head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize LM head weights."""
        if not self.config.tie_word_embeddings:
            std = self.config.initializer_range
            nn.init.normal_(self.lm_head.weight, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> GSACausalLMOutput:
        """
        Forward pass for causal LM.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            positions: Position indices
            attention_mask: Attention mask
            labels: Target labels for loss computation [batch, seq_len]
            past_key_values: KV cache
            use_cache: Whether to return updated cache
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states

        Returns:
            GSACausalLMOutput with loss, logits, and optional outputs
        """
        # Forward through base model
        outputs = self.model(
            input_ids=input_ids,
            positions=positions,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # LM head
        logits = self.lm_head(outputs.last_hidden_state)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Cross entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return GSACausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            do_sample: Whether to sample (vs greedy)
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID

        Returns:
            Generated token IDs [batch, seq_len + new_tokens]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        past_key_values = None
        generated = input_ids

        for _ in range(max_new_tokens):
            # Forward pass
            if past_key_values is not None:
                # Only process last token with cache
                curr_input = generated[:, -1:]
            else:
                curr_input = generated

            outputs = self.forward(
                input_ids=curr_input,
                past_key_values=past_key_values,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]  # [batch, vocab]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample or greedy
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Append to generated
            generated = torch.cat([generated, next_token], dim=-1)

            # Check for EOS
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break

        return generated

    @classmethod
    def from_pretrained(cls, path: str) -> "GSALlamaForCausalLM":
        """Load pretrained model from path."""
        import json
        from pathlib import Path

        path = Path(path)

        # Load config
        with open(path / "config.json", "r") as f:
            config_dict = json.load(f)
        config = GSAConfig.from_dict(config_dict)

        # Create model
        model = cls(config)

        # Load weights
        from safetensors.torch import load_file
        if (path / "model.safetensors").exists():
            state_dict = load_file(path / "model.safetensors")
        else:
            state_dict = torch.load(path / "pytorch_model.bin", map_location="cpu")

        model.load_state_dict(state_dict)

        return model

    def save_pretrained(self, path: str):
        """Save model to path."""
        from pathlib import Path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save_pretrained(path)

        # Save weights
        from safetensors.torch import save_file
        save_file(self.state_dict(), path / "model.safetensors")


# Alias for convenience
GSAForCausalLM = GSALlamaForCausalLM
