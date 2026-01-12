# Dataset Recommendations for GSA Training

This guide provides recommendations for datasets to use when training and evaluating Gated Sparse Attention (GSA) models.

## Pre-training Datasets

### Recommended: SlimPajama (Primary Choice)

**SlimPajama-627B** is the recommended dataset for GSA pre-training.

- **Size**: 627 billion tokens
- **Source**: Cleaned and deduplicated version of RedPajama
- **Quality**: High quality, extensively filtered
- **Diversity**: Multiple domains (web, books, code, etc.)

```bash
# Download using HuggingFace datasets
python data/prepare_data.py download --dataset slimpajama --output_dir data/slimpajama

# Or directly in Python
from datasets import load_dataset
dataset = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)
```

**Why SlimPajama?**
1. Well-balanced domain distribution
2. High quality filtering removes noise
3. Deduplicated to reduce memorization
4. Suitable size for academic research (not too large, not too small)

### Alternative: RedPajama-v2

For larger scale experiments:

- **Size**: 30+ trillion tokens
- **Source**: Web crawls, books, code, etc.
- **Use case**: Very large scale pre-training

```python
from datasets import load_dataset
dataset = load_dataset("togethercomputer/RedPajama-Data-V2", split="train")
```

### Alternative: The Pile

For diverse domain coverage:

- **Size**: 825 GB (approx. 300B tokens)
- **Domains**: 22 diverse subsets
- **Use case**: Domain-diverse pre-training

```python
from datasets import load_dataset
dataset = load_dataset("EleutherAI/pile", split="train")
```

### Alternative: FineWeb

For web-focused pre-training:

- **Size**: 15+ trillion tokens
- **Quality**: High-quality web data
- **Source**: CommonCrawl with extensive filtering

```python
from datasets import load_dataset
dataset = load_dataset("HuggingFaceFW/fineweb", split="train")
```

## Training Data Preparation

### Step 1: Download Data

```bash
# For SlimPajama (recommended)
python data/prepare_data.py download \
    --dataset slimpajama \
    --output_dir data/raw/slimpajama

# For a smaller test run (10M samples)
python data/prepare_data.py download \
    --dataset slimpajama \
    --output_dir data/raw/slimpajama_small \
    --num_samples 10000000
```

### Step 2: Tokenize Data

```bash
python data/prepare_data.py tokenize \
    --input_dir data/raw/slimpajama \
    --output_dir data/tokenized \
    --tokenizer meta-llama/Llama-3.1-8B \
    --max_seq_len 4096 \
    --num_workers 16
```

### Step 3: Pack Sequences

```bash
python data/prepare_data.py pack \
    --input_dir data/tokenized \
    --output_dir data/packed \
    --max_seq_len 4096
```

## Evaluation Datasets

### Language Modeling

| Dataset | Purpose | Size | Download |
|---------|---------|------|----------|
| WikiText-103 | Perplexity evaluation | 100M tokens | `load_dataset("wikitext", "wikitext-103-raw-v1")` |
| C4 | Large-scale PPL | 365B tokens | `load_dataset("allenai/c4", "en")` |
| LAMBADA | Word prediction | 10K examples | `load_dataset("lambada")` |

### Downstream Tasks

| Dataset | Task | Size | Download |
|---------|------|------|----------|
| MMLU | Knowledge/reasoning | 14K questions | `load_dataset("cais/mmlu", "all")` |
| GSM8K | Math reasoning | 8.5K problems | `load_dataset("gsm8k", "main")` |
| HumanEval | Code generation | 164 problems | `load_dataset("openai_humaneval")` |
| HellaSwag | Common sense | 10K questions | `load_dataset("hellaswag")` |
| ARC | Science QA | 7.7K questions | `load_dataset("ai2_arc", "ARC-Challenge")` |
| TruthfulQA | Truthfulness | 817 questions | `load_dataset("truthful_qa", "generation")` |
| WinoGrande | Coreference | 44K questions | `load_dataset("winogrande", "winogrande_xl")` |

### Long Context Evaluation

| Dataset | Purpose | Lengths | Download |
|---------|---------|---------|----------|
| RULER | Length generalization | 4K-128K | Custom generation (see `evaluate_ruler`) |
| Needle-in-Haystack | Information retrieval | Variable | Custom generation |
| LongBench | Long context QA | 4K-32K | `load_dataset("THUDM/LongBench")` |
| InfiniteBench | Extended context | 128K+ | `load_dataset("xinrongzhang2022/InfiniteBench")` |

## Quick Start Script

```python
#!/usr/bin/env python3
"""
Quick setup script for GSA training data.
"""

from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_small_dataset():
    """Prepare a small dataset for testing."""

    # Load a small subset of SlimPajama
    print("Loading dataset...")
    dataset = load_dataset(
        "cerebras/SlimPajama-627B",
        split="train",
        streaming=True,
    ).take(100000)  # 100K examples for testing

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize
    print("Tokenizing...")
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=4096,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True)

    print("Done! Dataset ready for training.")
    return tokenized

if __name__ == "__main__":
    prepare_small_dataset()
```

## Resource Requirements

### Storage Requirements

| Dataset | Raw Size | Tokenized Size |
|---------|----------|----------------|
| SlimPajama-627B | ~1.2 TB | ~2.5 TB |
| SlimPajama-100B subset | ~200 GB | ~400 GB |
| SlimPajama-10B subset | ~20 GB | ~40 GB |
| WikiText-103 | 500 MB | 1 GB |

### Recommended Setup by Compute Budget

| Budget | Dataset Size | Expected Quality |
|--------|--------------|------------------|
| 1 GPU, 1 week | 10-50B tokens | Proof of concept |
| 8 GPUs, 1 week | 100-200B tokens | Good quality |
| 8 GPUs, 2-4 weeks | 400-600B tokens | Strong performance |
| 64+ GPUs, 2+ weeks | 1T+ tokens | State-of-the-art |

## Data Quality Guidelines

### Filtering Recommendations

1. **Remove duplicates**: Use MinHash or exact deduplication
2. **Language filtering**: Keep only target language(s)
3. **Quality filtering**: Remove very short or very long documents
4. **Toxicity filtering**: Apply toxicity classifiers
5. **PII removal**: Remove personal identifiable information

### Sequence Packing

For efficient training, always pack sequences:

```python
# Bad: Padding wastes compute
# [tokens, tokens, PAD, PAD, PAD, PAD, ...]

# Good: Packing maximizes efficiency
# [doc1_tokens, EOS, doc2_tokens, EOS, doc3_tokens, ...]
```

## Evaluation Best Practices

1. **Use multiple benchmarks**: No single benchmark captures all capabilities
2. **Report standard deviations**: Run multiple seeds for reliable results
3. **Include perplexity**: Always report perplexity as a baseline metric
4. **Test at multiple lengths**: For long-context models, test at 4K, 8K, 16K, 32K, etc.
5. **Compare fairly**: Match training tokens and model size for comparisons

## Citations

If you use these datasets, please cite the original sources:

```bibtex
@article{slimpajama2023,
  title={SlimPajama: A 627B token cleaned and deduplicated version of RedPajama},
  author={Cerebras},
  year={2023},
}

@article{together2023redpajama,
  title={RedPajama-Data: An Open Source Recipe to Reproduce LLaMA training dataset},
  author={Together Computer},
  year={2023},
}

@article{gao2020pile,
  title={The Pile: An 800GB Dataset of Diverse Text for Language Modeling},
  author={Gao, Leo and others},
  journal={arXiv preprint arXiv:2101.00027},
  year={2020},
}
```
