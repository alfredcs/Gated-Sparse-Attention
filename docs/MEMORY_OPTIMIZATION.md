# Memory Optimization Guide

## Issue Analysis

The original aggressive settings caused OOM:
- **Old settings**: `micro_batch_size=1`, `max_seq_len=1024` → ~35GB memory usage
- **Too aggressive**: `micro_batch_size=8`, `max_seq_len=2048` → Tried to allocate 51+ GB (OOM!)
- **New balanced**: `micro_batch_size=2`, `max_seq_len=1024` → Should use ~38-40GB (~60-70% GPU)

## Memory Breakdown

On 44GB GPU with this model:
- **Model weights**: ~7-8 GB (512M parameters)
- **Optimizer state** (AdamW): ~14-16 GB (2x model weights)
- **Activations**: ~12-15 GB per micro_batch with seq_len=1024
- **Other overhead**: ~2-3 GB

Total: ~35-42 GB depending on batch size

## Current Configuration

```yaml
micro_batch_size: 2              # 2x increase from original
gradient_accumulation_steps: 32  # Maintains global_batch_size=256
max_seq_len: 1024                # Kept same to avoid OOM
```

**Expected improvements:**
- GPU utilization: 22% → 44-50%
- Training speed: 2x faster per step
- Memory usage: ~38-40 GB (safe margin under 44GB)

## How to Run

**Correct syntax** (note the environment variable name and usage):

```bash
# Option 1: Export then run
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=4 training/train.py \
    --config training/configs/pretrain_512m_low_memory.yaml \
    --output_dir outputs/gsa-512m \
    --resume_from outputs/gsa-512m/step_5000

# Option 2: Inline (single line)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=4 training/train.py --config training/configs/pretrain_512m_low_memory.yaml --output_dir outputs/gsa-512m --resume_from outputs/gsa-512m/step_5000
```

## Progressive Tuning Strategy

If you want to push GPU utilization higher, try these settings progressively:

### 1. Current (Safe - 60-70% GPU)
```yaml
micro_batch_size: 2
gradient_accumulation_steps: 32
max_seq_len: 1024
```

### 2. Medium (Target - 70-80% GPU)
```yaml
micro_batch_size: 3
gradient_accumulation_steps: 21  # 3 * 21 * 4 = 252 (close to 256)
max_seq_len: 1024
```

### 3. Aggressive (Risky - 80-90% GPU)
```yaml
micro_batch_size: 4
gradient_accumulation_steps: 16
max_seq_len: 1024
```

### 4. Maximum (Very Risky - May OOM)
```yaml
micro_batch_size: 2
gradient_accumulation_steps: 32
max_seq_len: 1536  # Increased sequence length
```

## Memory Saving Tips

If you still hit OOM, try these additional techniques:

### 1. Enable Gradient Checkpointing (Already enabled)
```yaml
gradient_checkpointing: true  # ✓ Already in config
```

### 2. Use Activation Checkpointing More Aggressively
Add to your model config if available.

### 3. Reduce Model Size
```yaml
model:
  n_layers: 6        # Reduce from 8
  d_ffn: 1024       # Reduce from 1536
```

### 4. Use FP16 Instead of BF16 (with GradScaler)
```yaml
precision: fp16  # Slightly more memory efficient than bf16
```

### 5. Reduce K Values
```yaml
gsa:
  k_base: 256      # Reduce from 512
  k_max: 512       # Reduce from 1024
```

## Monitoring Memory Usage

While training, monitor GPU memory on another terminal:

```bash
# Watch GPU memory every 2 seconds
watch -n 2 nvidia-smi

# Or continuous logging
nvidia-smi dmon -s mu -d 2
```

## Troubleshooting

### If OOM persists with current settings:
1. Reduce `micro_batch_size` back to 1
2. Keep `gradient_accumulation_steps` at 64
3. Monitor memory and gradually increase from there

### If GPU utilization is still low:
1. Check if data loading is a bottleneck (increase `num_workers`)
2. Verify disk I/O is not limiting throughput
3. Try the "Medium" settings above

## Expected Performance

With `micro_batch_size=2`:
- **Steps/iteration**: Same as before
- **Time per step**: ~50% of original (~27s vs 54s)
- **Total training time**: Cut in half
- **GPU utilization**: 44-50% (2x improvement)
- **Memory usage**: ~38-40 GB (safe under 44GB limit)

## Next Steps

1. Start with current safe settings
2. Monitor memory usage during first 100 steps
3. If memory stays under 40GB consistently, try "Medium" settings
4. If memory approaches 43GB, stay with current settings
