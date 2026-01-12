from gsa import GSAConfig, GSAForCausalLM
import torch

# Model config (memory-efficient: ~2.5GB for <48GB GPU)
config = GSAConfig(
    d_model=1024,          # Reduced model size
    n_layers=12,           # Fewer layers
    n_heads=8,
    n_kv_heads=2,
    d_ffn=2752,            # Reduced FFN size
    vocab_size=128256,
    k_base=256,            # Smaller sparse attention
    use_value_gate=True,
    use_output_gate=True,
)

# Create model
model = GSAForCausalLM(config).cuda()

# Forward pass with smaller batch
input_ids = torch.randint(0, 128256, (1, 256), device='cuda')  # batch=1, seq=256
outputs = model(input_ids, labels=input_ids)

print(f"Loss: {outputs.loss.item():.4f}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")