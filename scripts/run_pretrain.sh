#!/bin/bash
# Run GSA pre-training

set -e

# Configuration
CONFIG="${CONFIG:-training/configs/pretrain_1b.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gsa-1b-pretrain}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Environment setup
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_PROJECT="${WANDB_PROJECT:-gsa-pretrain}"
export OMP_NUM_THREADS=8

echo "============================================"
echo "GSA Pre-training Script"
echo "============================================"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "============================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Launch training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    training/train.py \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --wandb_run_name "gsa-1b-$(date +%Y%m%d-%H%M%S)" \
    "$@"

echo "============================================"
echo "Training Complete!"
echo "============================================"
echo "Checkpoints saved to: $OUTPUT_DIR"
