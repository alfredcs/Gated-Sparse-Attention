# GSA Quick Start Guide

Get up and running with Gated Sparse Attention in 5 minutes.

**Quick Links:**
- [Main README](README.md) - Project overview and benchmark results
- [Development Guide](README-development.md) - Architecture details, training, and evaluation

---

## Installation

```bash
cd gated-sparse-attention

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install package
pip install -e ".[all]"
```

## Verify Installation

```bash
# Run tests
pytest tests/ -v -x

# Quick import test
python -c "from gsa import GSAConfig, GatedSparseAttention; print('GSA installed successfully!')"
```

## GPU Memory Requirements

Choose model size based on your GPU:

| GPU Memory | Model Config | Parameters | Recommended Use |
|------------|--------------|------------|-----------------|
| 16-24 GB   | d_model=512, n_layers=6, k_base=128 | ~300M | Testing, small experiments |
| 24-40 GB   | d_model=1024, n_layers=12, k_base=256 | ~1B | Development, medium models |
| 40-80 GB   | d_model=2048, n_layers=24, k_base=512 | ~3-7B | Full training, large models |
| 80+ GB     | d_model=4096, n_layers=32, k_base=1024 | 13B+ | Production models |

**Before running examples:**
```bash
# Check available GPU memory
nvidia-smi

# Set memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Basic Usage

### 1. Create a GSA Attention Layer

```python
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
```

**For larger GPUs (80GB+), use these settings:**
```python
config = GSAConfig(
    d_model=2048,
    n_heads=16,
    n_kv_heads=4,
    k_base=512,
    use_value_gate=True,
    use_output_gate=True,
)
x = torch.randn(2, 1024, 2048, device='cuda')
positions = torch.arange(1024, device='cuda').unsqueeze(0).expand(2, -1)
```

### 2. Create a Full GSA Model

```python
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
```

**For larger GPUs (80GB+):**
```python
config = GSAConfig(
    d_model=2048,
    n_layers=24,
    n_heads=16,
    n_kv_heads=4,
    d_ffn=5504,
    vocab_size=128256,
    k_base=512,
    use_value_gate=True,
    use_output_gate=True,
)
input_ids = torch.randint(0, 128256, (2, 512), device='cuda')
```

### 3. Generate Text

```python
from transformers import AutoTokenizer

# Use Qwen2.5-7B tokenizer (32K context, efficient)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)

inputs = tokenizer("The future of AI is", return_tensors="pt").to("cuda")
outputs = model.generate(
    inputs["input_ids"],
    max_new_tokens=50,
    temperature=0.7,
    do_sample=True,
)
print(tokenizer.decode(outputs[0]))
```

**Alternative tokenizers:**
- LLaMA 3.1: `meta-llama/Llama-3.1-8B` (128K context, may hit memory limits)
- Qwen2.5: `Qwen/Qwen2.5-7B` (32K context, recommended)
- Mistral: `mistralai/Mistral-7B-v0.1` (32K context)

## Training

### 1. Prepare Data

```bash
# Download small dataset for testing (1M samples)
python data/prepare_data.py download \
    --dataset slimpajama \
    --output_dir data/raw \
    --num_samples 1000000

# Tokenize (using Qwen2.5-7B with 32K context)
python data/prepare_data.py tokenize \
    --input_dir data/raw \
    --output_dir data/tokenized \
    --tokenizer Qwen/Qwen2.5-7B \
    --max_seq_len 32768 \
    --num_workers 4

# Pack into sequences
python data/prepare_data.py pack \
    --input_dir data/tokenized \
    --output_dir data/packed \
    --max_seq_len 4096
```

### 2. Run Training

**Important:** The training script expects local packed data in `data/packed/train.bin`. Make sure you've completed step 1 (Prepare Data) first!

```bash
# Single GPU (for testing)
python training/train.py \
    --config training/configs/pretrain_1b.yaml \
    --output_dir outputs/gsa-test

# Multi-GPU (4 GPUs, optimized for 44GB GPU)
export PYTORCH_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=4 training/train.py \
    --config training/configs/pretrain_1b.yaml \
    --output_dir outputs/gsa-1b
```

**Note:** The config is optimized for 4x 44GB GPUs. For different hardware:
- Adjust `micro_batch_size` in the config (lower = less memory)
- Adjust `max_seq_len` (2048 → 1024 for even less memory)
- Adjust `--nproc_per_node` to match your GPU count

### 3. Training Config

Edit `training/configs/pretrain_1b.yaml`:

```yaml
model:
  d_model: 1024      # Model dimension (1024 for <48GB GPU)
  n_layers: 16       # Number of layers
  n_heads: 8
  n_kv_heads: 2
  vocab_size: 151936 # Qwen2.5 vocab size
  gsa:
    k_base: 1024         # Tokens to select (adjust for speed/quality)
    k_min: 128
    k_max: 2048
    use_value_gate: true
    use_output_gate: true
    use_adaptive_k: true

training:
  learning_rate: 3.0e-4
  max_steps: 100000
  micro_batch_size: 1         # Batch per GPU (reduce if OOM)
  gradient_accumulation_steps: 16
  max_seq_len: 2048           # Sequence length (reduce if OOM)
  precision: bf16
  gradient_checkpointing: true

data:
  data_dir: data/packed       # Local packed data directory
  tokenizer: Qwen/Qwen2.5-7B
  num_workers: 4
```

## Evaluation

### Download Eval Datasets

```bash
bash scripts/download_data.sh
```

### Run Evaluation

```bash
python evaluation/evaluate.py \
    --model_path outputs/gsa-1b/final \
    --benchmarks perplexity mmlu hellaswag \
    --output_dir results
```

## Convert Existing Models

Replace attention in a pre-trained LLaMA model:

```python
from gsa import GSAConfig, replace_attention_with_gsa
from transformers import LlamaForCausalLM

# Load pre-trained model
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

# Replace with GSA
gsa_config = GSAConfig(k_base=512, use_value_gate=True, use_output_gate=True)
model = replace_attention_with_gsa(model, gsa_config)

# Continue training or use directly
```

## Key Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `k_base` | Tokens to select per query | 512-2048 |
| `k_min` | Minimum tokens | 128-256 |
| `k_max` | Maximum tokens | 2048-4096 |
| `use_value_gate` | Enable G2 gate | `True` |
| `use_output_gate` | Enable G1 gate | `True` |
| `use_adaptive_k` | Dynamic sparsity | `True` |
| `n_indexer_heads` | Indexer heads | 4 |
| `d_indexer` | Indexer dimension | 64 |

## Performance Tips

1. **Start with smaller k_base** (512) for faster iteration, increase for better quality
2. **Enable both gates** (G1 and G2) for best results
3. **Use bf16 precision** for training stability
4. **Enable gradient checkpointing** for longer sequences

## Common Issues

### CUDA Out of Memory

**Quick fixes (try in order):**

1. **Clear GPU cache before running:**
```python
import torch
torch.cuda.empty_cache()
```

2. **Set memory fragmentation fix:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

3. **Reduce model size:**
```python
# Use smaller dimensions
config = GSAConfig(
    d_model=512,        # Instead of 1024 or 2048
    n_layers=6,         # Instead of 12 or 24
    n_heads=8,          # Instead of 16
    k_base=128,         # Instead of 256 or 512
)
```

4. **Reduce batch size and sequence length:**
```python
# For inference/testing
x = torch.randn(1, 256, 512, device='cuda')  # batch=1, seq=256, dim=512

# For training
config.training.micro_batch_size = 1
config.training.max_seq_len = 1024
```

5. **Use gradient checkpointing (for training):**
```python
model.gradient_checkpointing_enable()
```

6. **Monitor memory usage:**
```python
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### Slow Training
```python
# Reduce k_base for faster attention
gsa_config.k_base = 128  # or 256
# Reduce sequence length
config.training.max_seq_len = 2048
```

### Import Errors
```bash
# Ensure you're in the project directory
cd gated-sparse-attention
pip install -e .
```

## Project Structure

```
gated-sparse-attention/
├── gsa/                    # Core package
│   ├── attention/          # GSA components
│   ├── models/             # Full models
│   └── kernels/            # Triton kernels
├── training/               # Training code
├── evaluation/             # Benchmarks
├── data/                   # Data prep
├── tests/                  # Tests
└── scripts/                # Helper scripts
```

## Next Steps

1. Read `docs/datasets.md` for dataset recommendations
2. Check `training/configs/` for config options
3. Run `pytest tests/ -v` to verify everything works
4. Start with a small experiment before scaling up

## Links

- [Main README](README.md) - Project overview and benchmark results
- [Development Guide](README-development.md) - Architecture details and implementation
- Full documentation: `docs/`
- Training configs: `training/configs/`
- Benchmark code: `evaluation/benchmarks/`
