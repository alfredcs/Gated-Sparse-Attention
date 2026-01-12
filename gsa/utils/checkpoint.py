"""
Checkpoint utilities for GSA models.
"""

import os
import json
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Union
from omegaconf import DictConfig, OmegaConf


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    step: int,
    path: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    extra_state: Optional[Dict[str, Any]] = None,
):
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: LR scheduler state
        step: Current training step
        path: Save path
        config: Model/training config
        extra_state: Additional state to save
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Prepare state dict
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if config is not None:
        # Convert OmegaConf DictConfig to regular dict to avoid serialization issues
        if isinstance(config, DictConfig):
            checkpoint["config"] = OmegaConf.to_container(config, resolve=True)
        else:
            checkpoint["config"] = config

    if extra_state is not None:
        checkpoint["extra_state"] = extra_state

    # Save checkpoint
    checkpoint_path = path / f"checkpoint_{step}.pt"
    torch.save(checkpoint, checkpoint_path)

    # Save latest symlink
    latest_path = path / "checkpoint_latest.pt"
    if latest_path.exists():
        latest_path.unlink()
    latest_path.symlink_to(checkpoint_path.name)

    # Save config separately for easy access
    if config is not None:
        # Convert OmegaConf DictConfig to regular dict for JSON serialization
        if isinstance(config, DictConfig):
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = config
        with open(path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        path: Checkpoint path or directory
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        strict: Whether to strictly enforce state dict matching
        map_location: Device to map tensors to

    Returns:
        Dictionary with step and any extra state
    """
    path = Path(path)

    # Find checkpoint file
    if path.is_dir():
        checkpoint_path = path / "checkpoint_latest.pt"
        if not checkpoint_path.exists():
            # Find most recent checkpoint
            checkpoints = sorted(path.glob("checkpoint_*.pt"))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {path}")
            checkpoint_path = checkpoints[-1]
    else:
        checkpoint_path = path

    print(f"Loading checkpoint from {checkpoint_path}")

    # Load checkpoint
    # Note: weights_only=False is required to load optimizer/scheduler states
    # Only load checkpoints from trusted sources
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "step": checkpoint.get("step", 0),
        "config": checkpoint.get("config"),
        "extra_state": checkpoint.get("extra_state"),
    }


def convert_hf_checkpoint(
    hf_model_path: str,
    output_path: str,
    gsa_config: Optional[Dict[str, Any]] = None,
):
    """
    Convert HuggingFace model checkpoint to GSA format.

    Args:
        hf_model_path: Path to HuggingFace model
        output_path: Output path for GSA model
        gsa_config: GSA-specific configuration
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    # Load HF model
    print(f"Loading HuggingFace model from {hf_model_path}")
    hf_config = AutoConfig.from_pretrained(hf_model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path)

    # Create GSA config
    from ..config import GSAConfig

    gsa_cfg = GSAConfig(
        d_model=hf_config.hidden_size,
        n_heads=hf_config.num_attention_heads,
        n_kv_heads=getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
        n_layers=hf_config.num_hidden_layers,
        d_ffn=hf_config.intermediate_size,
        vocab_size=hf_config.vocab_size,
        max_position_embeddings=hf_config.max_position_embeddings,
        **(gsa_config or {}),
    )

    # Create GSA model
    from ..models import GSAForCausalLM
    gsa_model = GSAForCausalLM(gsa_cfg)

    # Map weights (this is model-specific, shown for LLaMA)
    state_dict = {}
    hf_state_dict = hf_model.state_dict()

    # Embeddings
    state_dict["model.embed_tokens.weight"] = hf_state_dict["model.embed_tokens.weight"]

    # LM head
    if "lm_head.weight" in hf_state_dict:
        state_dict["lm_head.weight"] = hf_state_dict["lm_head.weight"]

    # Final norm
    state_dict["model.norm.weight"] = hf_state_dict["model.norm.weight"]

    # Layers
    for i in range(gsa_cfg.n_layers):
        prefix = f"model.layers.{i}"
        hf_prefix = f"model.layers.{i}"

        # Input layernorm
        state_dict[f"{prefix}.input_layernorm.weight"] = hf_state_dict[f"{hf_prefix}.input_layernorm.weight"]

        # Self attention (Q, K, V, O projections)
        state_dict[f"{prefix}.self_attn.q_proj.weight"] = hf_state_dict[f"{hf_prefix}.self_attn.q_proj.weight"]
        state_dict[f"{prefix}.self_attn.k_proj.weight"] = hf_state_dict[f"{hf_prefix}.self_attn.k_proj.weight"]
        state_dict[f"{prefix}.self_attn.v_proj.weight"] = hf_state_dict[f"{hf_prefix}.self_attn.v_proj.weight"]
        state_dict[f"{prefix}.self_attn.o_proj.weight"] = hf_state_dict[f"{hf_prefix}.self_attn.o_proj.weight"]

        # Post attention layernorm
        state_dict[f"{prefix}.post_attention_layernorm.weight"] = hf_state_dict[f"{hf_prefix}.post_attention_layernorm.weight"]

        # MLP
        state_dict[f"{prefix}.mlp.gate_proj.weight"] = hf_state_dict[f"{hf_prefix}.mlp.gate_proj.weight"]
        state_dict[f"{prefix}.mlp.up_proj.weight"] = hf_state_dict[f"{hf_prefix}.mlp.up_proj.weight"]
        state_dict[f"{prefix}.mlp.down_proj.weight"] = hf_state_dict[f"{hf_prefix}.mlp.down_proj.weight"]

    # Initialize GSA-specific weights randomly
    # (indexer, gates) - these need training
    gsa_state_dict = gsa_model.state_dict()
    for key in gsa_state_dict:
        if key not in state_dict:
            state_dict[key] = gsa_state_dict[key]
            print(f"Initialized {key} randomly")

    # Load state dict
    gsa_model.load_state_dict(state_dict, strict=False)

    # Save
    gsa_model.save_pretrained(output_path)
    print(f"Saved GSA model to {output_path}")
