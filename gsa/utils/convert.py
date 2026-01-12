"""
Conversion utilities for replacing attention with GSA.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, List

from ..config import GSAConfig
from ..attention import GatedSparseAttention


def replace_attention_with_gsa(
    model: nn.Module,
    config: GSAConfig,
    layers_to_replace: Union[str, List[int]] = "all",
    preserve_weights: bool = True,
) -> nn.Module:
    """
    Replace standard attention layers with GSA in an existing model.

    Args:
        model: Pre-trained model (e.g., LlamaForCausalLM)
        config: GSA configuration
        layers_to_replace: "all" or list of layer indices
        preserve_weights: Whether to copy existing Q, K, V, O weights

    Returns:
        Modified model with GSA attention
    """

    def _get_attention_layers(module, prefix=""):
        """Find all attention layers in the model."""
        layers = {}
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if _is_attention_layer(child):
                layers[full_name] = child
            else:
                layers.update(_get_attention_layers(child, full_name))
        return layers

    def _is_attention_layer(module):
        """Check if module is an attention layer."""
        module_name = module.__class__.__name__.lower()
        return any(x in module_name for x in ["attention", "attn"])

    def _create_gsa_layer(original_attn, config):
        """Create GSA layer and optionally copy weights."""
        gsa = GatedSparseAttention(config)

        if preserve_weights:
            # Copy Q, K, V, O projection weights
            if hasattr(original_attn, "q_proj"):
                gsa.q_proj.weight.data.copy_(original_attn.q_proj.weight.data)
            if hasattr(original_attn, "k_proj"):
                gsa.k_proj.weight.data.copy_(original_attn.k_proj.weight.data)
            if hasattr(original_attn, "v_proj"):
                gsa.v_proj.weight.data.copy_(original_attn.v_proj.weight.data)
            if hasattr(original_attn, "o_proj"):
                gsa.o_proj.weight.data.copy_(original_attn.o_proj.weight.data)

        return gsa

    def _replace_module(parent, name, new_module):
        """Replace a module in the parent."""
        setattr(parent, name, new_module)

    # Find all attention layers
    attention_layers = _get_attention_layers(model)

    # Determine which layers to replace
    if layers_to_replace == "all":
        layers_to_replace_set = set(attention_layers.keys())
    else:
        layers_to_replace_set = set()
        for idx in layers_to_replace:
            # Find layer by index
            for name in attention_layers:
                if f".{idx}." in name or name.endswith(f".{idx}"):
                    layers_to_replace_set.add(name)

    # Replace attention layers
    for layer_name in layers_to_replace_set:
        original_attn = attention_layers[layer_name]
        gsa = _create_gsa_layer(original_attn, config)

        # Navigate to parent and replace
        parts = layer_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        _replace_module(parent, parts[-1], gsa)
        print(f"Replaced {layer_name} with GSA")

    return model


def convert_llama_to_gsa(
    model_name_or_path: str,
    gsa_config: Optional[GSAConfig] = None,
    output_path: Optional[str] = None,
) -> nn.Module:
    """
    Convert a LLaMA model to use GSA attention.

    Args:
        model_name_or_path: HuggingFace model name or local path
        gsa_config: GSA configuration (auto-generated if None)
        output_path: Optional path to save converted model

    Returns:
        Converted model with GSA attention
    """
    from transformers import LlamaForCausalLM, LlamaConfig

    # Load original model
    print(f"Loading {model_name_or_path}...")
    original_model = LlamaForCausalLM.from_pretrained(model_name_or_path)
    original_config = original_model.config

    # Create GSA config if not provided
    if gsa_config is None:
        gsa_config = GSAConfig(
            d_model=original_config.hidden_size,
            n_heads=original_config.num_attention_heads,
            n_kv_heads=getattr(original_config, "num_key_value_heads", original_config.num_attention_heads),
            n_layers=original_config.num_hidden_layers,
            d_ffn=original_config.intermediate_size,
            vocab_size=original_config.vocab_size,
            max_position_embeddings=original_config.max_position_embeddings,
        )

    # Replace attention layers
    converted_model = replace_attention_with_gsa(
        original_model,
        gsa_config,
        layers_to_replace="all",
        preserve_weights=True,
    )

    # Save if path provided
    if output_path:
        from pathlib import Path
        path = Path(output_path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(converted_model.state_dict(), path / "pytorch_model.bin")

        # Save config
        import json
        with open(path / "config.json", "w") as f:
            json.dump(gsa_config.to_dict(), f, indent=2)

        print(f"Saved converted model to {output_path}")

    return converted_model
