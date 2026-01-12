import torch
from gsa import GSAConfig, GatedSparseAttention

# Configure GSA (memory-efficient settings for <48GB GPU)
config = GSAConfig(
    d_model=1024,          # Reduced from 2048
    n_heads=8,             # Reduced from 16
    n_kv_heads=2,          # GQA support
    k_base=256,            # Select top-256 tokens (reduced from 512)
    use_value_gate=True,   # Enable G2 gate
    use_output_gate=True,  # Enable G1 gate
)

# Create layer
gsa = GatedSparseAttention(config).cuda()

# Forward pass (small batch for testing)
x = torch.randn(1, 512, 1024, device='cuda')  # [batch=1, seq=512, dim=1024]
positions = torch.arange(512, device='cuda').unsqueeze(0)

output, _, _ = gsa(x, positions=positions)
print(f"Output shape: {output.shape}")  # [1, 512, 1024]