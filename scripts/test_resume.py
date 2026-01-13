#!/usr/bin/env python3
"""
Test script to verify checkpoint resume functionality.
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))

from gsa.utils import load_checkpoint
from gsa import GSAForCausalLM, GSAConfig

def test_checkpoint_resume():
    """Test that checkpoints can be loaded and contain the right state."""

    checkpoint_path = "outputs/gsa-512m/step_5000"

    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return False

    print(f"Testing checkpoint at: {checkpoint_path}")
    print("-" * 60)

    # Load checkpoint metadata without model
    ckpt = torch.load(
        f"{checkpoint_path}/checkpoint_5000.pt",
        map_location="cpu",
        weights_only=False
    )

    print(f"✓ Checkpoint loaded successfully")
    print(f"  Step: {ckpt['step']}")
    print(f"  Epoch: {ckpt.get('extra_state', {}).get('epoch', 'N/A')}")
    print(f"  Has model state: {'model_state_dict' in ckpt}")
    print(f"  Has optimizer state: {'optimizer_state_dict' in ckpt}")
    print(f"  Has scheduler state: {'scheduler_state_dict' in ckpt}")
    print(f"  Has config: {'config' in ckpt}")

    # Check config is a dict (not DictConfig)
    if 'config' in ckpt:
        config_type = type(ckpt['config']).__name__
        print(f"  Config type: {config_type}")
        if config_type == "dict":
            print(f"  ✓ Config is properly serialized as dict (new format)")
        elif config_type == "DictConfig":
            print(f"  ⚠ Config is DictConfig (old format, can still be loaded)")
            print(f"    New checkpoints will use dict format after the fix")
        else:
            print(f"  ❌ Config is {config_type}, unexpected type")
            return False

    # Try to create model from config and load state
    print("\nTesting model restoration...")
    try:
        if 'config' in ckpt and ckpt['config'] is not None:
            # Handle both dict and DictConfig
            from omegaconf import DictConfig, OmegaConf
            cfg = ckpt['config']
            if isinstance(cfg, DictConfig):
                cfg = OmegaConf.to_container(cfg, resolve=True)

            model_cfg = cfg.get('model', {})
            gsa_cfg = model_cfg.get('gsa', {})

            config = GSAConfig(
                d_model=model_cfg.get('d_model', 512),
                n_layers=model_cfg.get('n_layers', 8),
                n_heads=model_cfg.get('n_heads', 8),
                n_kv_heads=model_cfg.get('n_kv_heads', 2),
                d_ffn=model_cfg.get('d_ffn', 1536),
                vocab_size=model_cfg.get('vocab_size', 151936),
                max_position_embeddings=model_cfg.get('max_position_embeddings', 8192),
                d_indexer=gsa_cfg.get('d_indexer', 64),
                n_indexer_heads=gsa_cfg.get('n_indexer_heads', 4),
                k_base=gsa_cfg.get('k_base', 512),
                k_min=gsa_cfg.get('k_min', 64),
                k_max=gsa_cfg.get('k_max', 1024),
                use_value_gate=gsa_cfg.get('use_value_gate', True),
                use_output_gate=gsa_cfg.get('use_output_gate', True),
                use_adaptive_k=gsa_cfg.get('use_adaptive_k', True),
            )

            model = GSAForCausalLM(config)
            model.load_state_dict(ckpt['model_state_dict'], strict=True)
            print(f"  ✓ Model state loaded successfully")
        else:
            print(f"  ⚠ No config in checkpoint, skipping model load test")
    except Exception as e:
        print(f"  ❌ Failed to load model state: {e}")
        return False

    print("\n" + "="*60)
    print("✅ All checkpoint resume tests passed!")
    print("="*60)
    print("\nTo resume training, run:")
    print(f"torchrun --nproc_per_node=4 training/train.py \\")
    print(f"    --config training/configs/pretrain_512m_low_memory.yaml \\")
    print(f"    --output_dir outputs/gsa-512m \\")
    print(f"    --resume_from {checkpoint_path}")

    return True

if __name__ == "__main__":
    success = test_checkpoint_resume()
    sys.exit(0 if success else 1)
