#!/bin/bash
# Download datasets for GSA training and evaluation

set -e

DATA_DIR="${DATA_DIR:-data}"
CACHE_DIR="${CACHE_DIR:-$DATA_DIR/cache}"

echo "============================================"
echo "GSA Dataset Download Script"
echo "============================================"
echo "Data directory: $DATA_DIR"
echo "Cache directory: $CACHE_DIR"
echo ""

# Create directories
mkdir -p "$DATA_DIR/eval"
mkdir -p "$DATA_DIR/train"
mkdir -p "$CACHE_DIR"

# Function to download dataset
download_dataset() {
    local name=$1
    local cmd=$2
    echo "Downloading $name..."
    python -c "$cmd" || echo "Warning: Failed to download $name"
}

echo "============================================"
echo "Downloading Evaluation Datasets"
echo "============================================"

# WikiText-103
download_dataset "WikiText-103" "
from datasets import load_dataset
load_dataset('wikitext', 'wikitext-103-raw-v1', cache_dir='$CACHE_DIR/wikitext')
print('WikiText-103 downloaded successfully')
"

# MMLU
download_dataset "MMLU" "
from datasets import load_dataset
load_dataset('cais/mmlu', 'all', cache_dir='$CACHE_DIR/mmlu')
print('MMLU downloaded successfully')
"

# GSM8K
download_dataset "GSM8K" "
from datasets import load_dataset
load_dataset('gsm8k', 'main', cache_dir='$CACHE_DIR/gsm8k')
print('GSM8K downloaded successfully')
"

# HellaSwag
download_dataset "HellaSwag" "
from datasets import load_dataset
load_dataset('hellaswag', cache_dir='$CACHE_DIR/hellaswag')
print('HellaSwag downloaded successfully')
"

# ARC-Challenge
download_dataset "ARC-Challenge" "
from datasets import load_dataset
load_dataset('ai2_arc', 'ARC-Challenge', cache_dir='$CACHE_DIR/arc')
print('ARC-Challenge downloaded successfully')
"

# TruthfulQA
download_dataset "TruthfulQA" "
from datasets import load_dataset
load_dataset('truthful_qa', 'generation', cache_dir='$CACHE_DIR/truthfulqa')
print('TruthfulQA downloaded successfully')
"

# WinoGrande
download_dataset "WinoGrande" "
from datasets import load_dataset
load_dataset('winogrande', 'winogrande_xl', cache_dir='$CACHE_DIR/winogrande')
print('WinoGrande downloaded successfully')
"

echo ""
echo "============================================"
echo "Download Training Data (Optional)"
echo "============================================"
echo ""
echo "To download training data, run:"
echo ""
echo "  # SlimPajama (recommended, ~1.2TB)"
echo "  python data/prepare_data.py download --dataset slimpajama --output_dir $DATA_DIR/train/slimpajama"
echo ""
echo "  # Smaller subset for testing (~20GB)"
echo "  python data/prepare_data.py download --dataset slimpajama --output_dir $DATA_DIR/train/slimpajama_small --num_samples 10000000"
echo ""

echo "============================================"
echo "Dataset Download Complete!"
echo "============================================"
echo ""
echo "Datasets saved to: $CACHE_DIR"
echo ""
echo "To verify downloads, run:"
echo "  python -c \"from datasets import load_dataset; print(load_dataset('wikitext', 'wikitext-103-raw-v1', split='test'))\""
