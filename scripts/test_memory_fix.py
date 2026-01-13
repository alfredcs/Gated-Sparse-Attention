#!/usr/bin/env python3
"""
Memory-efficient test script for GSA
Designed to work on GPUs with 24-48GB memory
"""

import torch
from gsa import GSAConfig, GatedSparseAttention

def print_memory_stats():
    """Print current GPU memory usage"""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory - Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Total: {total:.2f} GB")

def main():
    print("=" * 60)
    print("Testing GSA with Memory-Efficient Configuration")
    print("=" * 60)

    # Clear any cached memory
    torch.cuda.empty_cache()

    print("\n1. Initial Memory State:")
    print_memory_stats()

    # Configure GSA with memory-efficient settings
    print("\n2. Creating GSA Layer with memory-efficient config...")
    config = GSAConfig(
        d_model=1024,          # Reduced from 2048
        n_heads=8,             # Reduced from 16
        n_kv_heads=2,          # GQA support
        k_base=256,            # Select top-256 tokens (reduced from 512)
        use_value_gate=True,   # Enable G2 gate
        use_output_gate=True,  # Enable G1 gate
    )

    gsa = GatedSparseAttention(config).cuda()
    print("   ✓ GSA layer created successfully")
    print_memory_stats()

    # Forward pass with small batch
    print("\n3. Running forward pass (batch=1, seq=512)...")
    x = torch.randn(1, 512, 1024, device='cuda')
    positions = torch.arange(512, device='cuda').unsqueeze(0)

    output, _, _ = gsa(x, positions=positions)
    print(f"   ✓ Forward pass completed")
    print(f"   ✓ Output shape: {output.shape}")
    print_memory_stats()

    # Clean up
    del x, positions, output, gsa
    torch.cuda.empty_cache()

    print("\n4. Final Memory State (after cleanup):")
    print_memory_stats()

    print("\n" + "=" * 60)
    print("SUCCESS! GSA is working with memory-efficient settings")
    print("=" * 60)
    print("\nTo use larger models, see the README-qs.md GPU Memory Requirements table")

if __name__ == "__main__":
    main()
