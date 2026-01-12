"""
Data loading utilities for GSA training.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Optional, Dict, Any, Iterator
from pathlib import Path


class PretrainingDataset(Dataset):
    """
    Dataset for language model pretraining.

    Supports streaming from HuggingFace datasets or local tokenized files.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str,
        max_seq_len: int = 4096,
        split: str = "train",
        cache_dir: Optional[str] = None,
        local_dataset_path: Optional[str] = None,
    ):
        """
        Initialize dataset.

        Args:
            dataset_name: HuggingFace dataset name or local path
            tokenizer_name: Tokenizer to use
            max_seq_len: Maximum sequence length
            split: Dataset split
            cache_dir: Cache directory for downloaded data
            local_dataset_path: Local path to dataset (avoids HF API calls)
        """
        from datasets import load_dataset
        from transformers import AutoTokenizer
        import glob

        self.max_seq_len = max_seq_len

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load dataset
        if local_dataset_path and Path(local_dataset_path).exists():
            # Load from local dataset directory (e.g., /data/hf_datasets/SlimPajama-627B)
            local_path = Path(local_dataset_path)
            # Find all .jsonl.zst files in the split directory
            data_files = glob.glob(str(local_path / split / "**/*.jsonl.zst"), recursive=True)
            if data_files:
                print(f"Loading {len(data_files)} data files from {local_path / split}")
                self.dataset = load_dataset("json", data_files=data_files, split="train")
            else:
                raise FileNotFoundError(f"No .jsonl.zst files found in {local_path / split}")
        elif Path(dataset_name).exists():
            # Local dataset file
            self.dataset = load_dataset("json", data_files=dataset_name, split=split)
        else:
            # HuggingFace dataset (will call API)
            self.dataset = load_dataset(
                dataset_name,
                split=split,
                cache_dir=cache_dir,
                streaming=False,  # Set to True for very large datasets
            )

        # Determine text column
        if "text" in self.dataset.column_names:
            self.text_column = "text"
        elif "content" in self.dataset.column_names:
            self.text_column = "content"
        else:
            self.text_column = self.dataset.column_names[0]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.dataset[idx][self.text_column]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


class PackedDataset(Dataset):
    """
    Dataset that packs multiple documents into single sequences.

    More efficient for training as it minimizes padding.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str,
        max_seq_len: int = 4096,
        split: str = "train",
        cache_dir: Optional[str] = None,
        num_proc: int = 8,
        local_dataset_path: Optional[str] = None,
    ):
        """
        Initialize packed dataset.

        Args:
            dataset_name: HuggingFace dataset name or local path
            tokenizer_name: Tokenizer to use
            max_seq_len: Maximum sequence length
            split: Dataset split
            cache_dir: Cache directory
            num_proc: Number of processes for tokenization
            local_dataset_path: Local path to dataset (avoids HF API calls)
        """
        from datasets import load_dataset
        from transformers import AutoTokenizer
        import glob

        self.max_seq_len = max_seq_len

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load and tokenize dataset
        if local_dataset_path and Path(local_dataset_path).exists():
            # Load from local dataset directory
            local_path = Path(local_dataset_path)
            data_files = glob.glob(str(local_path / split / "**/*.jsonl.zst"), recursive=True)
            if data_files:
                print(f"Loading {len(data_files)} data files from {local_path / split}")
                dataset = load_dataset("json", data_files=data_files, split="train")
            else:
                raise FileNotFoundError(f"No .jsonl.zst files found in {local_path / split}")
        else:
            # Load from HuggingFace (will call API)
            dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)

        # Determine text column
        if "text" in dataset.column_names:
            text_column = "text"
        elif "content" in dataset.column_names:
            text_column = "content"
        else:
            text_column = dataset.column_names[0]

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                truncation=False,
                padding=False,
            )

        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
        )

        # Pack sequences
        self.packed_data = self._pack_sequences(tokenized)

    def _pack_sequences(self, tokenized_dataset) -> list:
        """Pack tokenized sequences into fixed-length chunks."""
        all_input_ids = []

        # Concatenate all tokens
        buffer = []
        for example in tokenized_dataset:
            buffer.extend(example["input_ids"])
            buffer.append(self.tokenizer.eos_token_id)

            # Create chunks when buffer is large enough
            while len(buffer) >= self.max_seq_len:
                all_input_ids.append(buffer[:self.max_seq_len])
                buffer = buffer[self.max_seq_len:]

        return all_input_ids

    def __len__(self) -> int:
        return len(self.packed_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor(self.packed_data[idx], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


class LocalBinaryDataset(Dataset):
    """
    Dataset that loads from local pre-tokenized binary files.

    Expects data from prepare_data.py pack command:
    - train.bin: packed sequences as uint32 numpy array
    - metadata.json: metadata with num_sequences, seq_len, etc.
    """

    def __init__(
        self,
        data_dir: str,
        max_seq_len: int = 4096,
    ):
        """
        Initialize local binary dataset.

        Args:
            data_dir: Directory containing train.bin and metadata.json
            max_seq_len: Maximum sequence length
        """
        import json

        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            self.num_sequences = self.metadata["num_sequences"]
            self.seq_len = self.metadata["seq_len"]
        else:
            # If no metadata, try to infer from train.bin
            bin_file = self.data_dir / "train.bin"
            if not bin_file.exists():
                raise FileNotFoundError(f"No train.bin found in {data_dir}")

            # Load entire file to get size
            data = np.fromfile(bin_file, dtype=np.uint32)
            total_tokens = len(data)
            self.seq_len = max_seq_len
            self.num_sequences = total_tokens // self.seq_len
            print(f"Inferred {self.num_sequences} sequences of length {self.seq_len}")

        # Memory-map the binary file for efficient random access
        self.data_file = self.data_dir / "train.bin"
        self.data = np.memmap(self.data_file, dtype=np.uint32, mode='r')

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Calculate start and end positions
        start = idx * self.seq_len
        end = start + self.seq_len

        # Load sequence from memory-mapped file
        input_ids = torch.from_numpy(self.data[start:end].copy()).long()

        # Truncate to max_seq_len if the packed data has longer sequences
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


def create_dataloader(
    config: Dict[str, Any],
    batch_size: int,
    max_seq_len: int,
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """
    Create data loader for training.

    Args:
        config: Data configuration
        batch_size: Batch size per GPU
        max_seq_len: Maximum sequence length
        num_workers: Number of data loading workers
        rank: Process rank for distributed training
        world_size: Total number of processes

    Returns:
        DataLoader instance
    """
    dataset_name = config.get("dataset", "cerebras/SlimPajama-627B")
    tokenizer_name = config.get("tokenizer", "Qwen/Qwen2.5-7B")
    cache_dir = config.get("cache_dir", None)
    use_packing = config.get("use_packing", True)
    data_dir = config.get("data_dir", None)
    local_dataset_path = config.get("local_dataset_path", None)

    # Check if we should use local binary data (highest priority)
    if data_dir and Path(data_dir).exists():
        print(f"Loading data from local binary directory: {data_dir}")
        dataset = LocalBinaryDataset(
            data_dir=data_dir,
            max_seq_len=max_seq_len,
        )
    # Create dataset from local HuggingFace dataset or HF API
    elif use_packing:
        dataset = PackedDataset(
            dataset_name=dataset_name,
            tokenizer_name=tokenizer_name,
            max_seq_len=max_seq_len,
            split="train",
            cache_dir=cache_dir,
            local_dataset_path=local_dataset_path,
        )
    else:
        dataset = PretrainingDataset(
            dataset_name=dataset_name,
            tokenizer_name=tokenizer_name,
            max_seq_len=max_seq_len,
            split="train",
            cache_dir=cache_dir,
            local_dataset_path=local_dataset_path,
        )

    # Create sampler for distributed training
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader
