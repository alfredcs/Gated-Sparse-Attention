#!/usr/bin/env python3
"""
Data preparation scripts for GSA training.

Supports:
- SlimPajama download and preprocessing
- RedPajama download and preprocessing
- Custom dataset tokenization
- Data packing for efficient training
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm
import multiprocessing as mp


def download_slimpajama(
    output_dir: str,
    num_samples: Optional[int] = None,
    subset: str = "train",
):
    """
    Download and prepare SlimPajama dataset.

    SlimPajama is a cleaned and deduplicated version of RedPajama (627B tokens).
    Recommended for pre-training.

    Args:
        output_dir: Output directory
        num_samples: Number of samples (None for full dataset)
        subset: Dataset subset
    """
    from datasets import load_dataset

    print("Downloading SlimPajama dataset...")
    print("This is a large dataset and may take several hours.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset with streaming
    dataset = load_dataset(
        "cerebras/SlimPajama-627B",
        split=subset,
        streaming=True,
    )

    # Process and save
    batch_size = 10000
    batch = []
    batch_idx = 0

    for i, example in enumerate(tqdm(dataset, desc="Processing")):
        if num_samples and i >= num_samples:
            break

        batch.append({"text": example["text"]})

        if len(batch) >= batch_size:
            # Save batch
            batch_file = output_path / f"batch_{batch_idx:06d}.jsonl"
            with open(batch_file, "w") as f:
                for item in batch:
                    f.write(json.dumps(item) + "\n")
            batch = []
            batch_idx += 1

    # Save remaining
    if batch:
        batch_file = output_path / f"batch_{batch_idx:06d}.jsonl"
        with open(batch_file, "w") as f:
            for item in batch:
                f.write(json.dumps(item) + "\n")

    print(f"Saved {batch_idx + 1} batches to {output_dir}")


def _process_file_worker(args):
    """
    Worker function to process a single file.
    Must be at module level for multiprocessing to pickle it.
    """
    from transformers import AutoTokenizer
    import numpy as np

    input_file, output_file, tokenizer_name, max_seq_len = args

    # Load tokenizer in worker process
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_tokens = []

    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            text = data.get("text", "")
            if not text:
                continue

            # Tokenize with truncation disabled to get full length
            tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)

            # If single text exceeds max length, split it into chunks
            if len(tokens) > max_seq_len:
                # Split into chunks of max_seq_len
                for i in range(0, len(tokens), max_seq_len):
                    chunk = tokens[i:i + max_seq_len]
                    all_tokens.extend(chunk)
                    all_tokens.append(tokenizer.eos_token_id)
            else:
                all_tokens.extend(tokens)
                all_tokens.append(tokenizer.eos_token_id)

    # Save as binary
    tokens_array = np.array(all_tokens, dtype=np.uint32)
    tokens_array.tofile(output_file)

    return len(all_tokens)


def tokenize_dataset(
    input_dir: str,
    output_dir: str,
    tokenizer_name: str = "Qwen/Qwen2.5-7B",
    max_seq_len: int = 32768,
    num_workers: int = 8,
):
    """
    Tokenize a dataset using specified tokenizer.

    Args:
        input_dir: Directory with JSONL files
        output_dir: Output directory for tokenized data
        tokenizer_name: Tokenizer to use (default: Qwen2.5-7B with 32K context)
        max_seq_len: Maximum sequence length for chunking long texts
        num_workers: Number of worker processes
    """
    from transformers import AutoTokenizer
    import numpy as np

    print(f"Loading tokenizer: {tokenizer_name}")
    # Verify tokenizer can be loaded
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = len(tokenizer)
    model_max_length = getattr(tokenizer, 'model_max_length', max_seq_len)

    print(f"Tokenizer loaded successfully")
    print(f"  Vocab size: {vocab_size:,}")
    print(f"  Model max length: {model_max_length:,}")
    print(f"  Chunking at: {max_seq_len:,} tokens")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all input files
    input_files = sorted(input_path.glob("*.jsonl"))
    print(f"Found {len(input_files)} input files")

    # Prepare arguments for worker processes
    worker_args = [
        (input_file, output_path / f"{input_file.stem}.bin", tokenizer_name, max_seq_len)
        for input_file in input_files
    ]

    # Process files in parallel
    total_tokens = 0
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(_process_file_worker, worker_args),
            total=len(input_files),
            desc="Tokenizing"
        ))
        total_tokens = sum(results)

    print(f"Total tokens: {total_tokens:,}")
    print(f"Saved to {output_dir}")


def pack_sequences(
    input_dir: str,
    output_dir: str,
    max_seq_len: int = 4096,
    eos_token_id: int = 2,
):
    """
    Pack tokenized sequences into fixed-length chunks.

    This creates training-ready data with minimal padding.

    Args:
        input_dir: Directory with .bin token files
        output_dir: Output directory
        max_seq_len: Maximum sequence length
        eos_token_id: EOS token ID
    """
    import numpy as np

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load all tokens
    print("Loading tokenized data...")
    all_tokens = []
    for bin_file in sorted(input_path.glob("*.bin")):
        tokens = np.fromfile(bin_file, dtype=np.uint32)
        all_tokens.extend(tokens.tolist())

    print(f"Total tokens: {len(all_tokens):,}")

    # Pack into sequences
    print("Packing sequences...")
    sequences = []
    for i in range(0, len(all_tokens) - max_seq_len, max_seq_len):
        seq = all_tokens[i:i + max_seq_len]
        sequences.append(seq)

    print(f"Created {len(sequences):,} sequences of length {max_seq_len}")

    # Save as numpy array
    sequences_array = np.array(sequences, dtype=np.uint32)
    output_file = output_path / "train.bin"
    sequences_array.tofile(output_file)

    # Save metadata
    metadata = {
        "num_sequences": len(sequences),
        "seq_len": max_seq_len,
        "total_tokens": len(sequences) * max_seq_len,
        "dtype": "uint32",
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved packed data to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Data preparation for GSA")
    subparsers = parser.add_subparsers(dest="command")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download dataset")
    download_parser.add_argument("--dataset", type=str, default="slimpajama")
    download_parser.add_argument("--output_dir", type=str, required=True)
    download_parser.add_argument("--num_samples", type=int, default=None)

    # Tokenize command
    tokenize_parser = subparsers.add_parser("tokenize", help="Tokenize dataset")
    tokenize_parser.add_argument("--input_dir", type=str, required=True)
    tokenize_parser.add_argument("--output_dir", type=str, required=True)
    tokenize_parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-7B",
                                 help="Tokenizer to use (default: Qwen2.5-7B with 32K context)")
    tokenize_parser.add_argument("--max_seq_len", type=int, default=32768,
                                 help="Max sequence length for chunking (default: 32768)")
    tokenize_parser.add_argument("--num_workers", type=int, default=8)

    # Pack command
    pack_parser = subparsers.add_parser("pack", help="Pack sequences")
    pack_parser.add_argument("--input_dir", type=str, required=True)
    pack_parser.add_argument("--output_dir", type=str, required=True)
    pack_parser.add_argument("--max_seq_len", type=int, default=4096)

    args = parser.parse_args()

    if args.command == "download":
        if args.dataset == "slimpajama":
            download_slimpajama(args.output_dir, args.num_samples)
    elif args.command == "tokenize":
        tokenize_dataset(
            args.input_dir,
            args.output_dir,
            args.tokenizer,
            args.max_seq_len,
            args.num_workers,
        )
    elif args.command == "pack":
        pack_sequences(args.input_dir, args.output_dir, args.max_seq_len)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
