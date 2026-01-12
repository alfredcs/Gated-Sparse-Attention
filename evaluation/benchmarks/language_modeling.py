"""
Language modeling evaluation (perplexity).
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from tqdm import tqdm
import math


def evaluate_perplexity(
    model: torch.nn.Module,
    config: Dict[str, Any],
    batch_size: int = 8,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate perplexity on language modeling datasets.

    Args:
        model: Model to evaluate
        config: Evaluation configuration
        batch_size: Batch size
        max_samples: Maximum samples to evaluate

    Returns:
        Dictionary with perplexity scores
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    model.eval()
    device = next(model.parameters()).device

    # Get tokenizer
    tokenizer_name = config.get("tokenizer", "meta-llama/Llama-3.1-8B")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    # Evaluate on multiple datasets
    datasets_to_eval = config.get("datasets", ["wikitext"])

    for dataset_name in datasets_to_eval:
        print(f"Evaluating perplexity on {dataset_name}...")

        # Load dataset
        if dataset_name == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
            text_column = "text"
        elif dataset_name == "c4":
            dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
            text_column = "text"
        else:
            dataset = load_dataset(dataset_name, split="test")
            text_column = "text" if "text" in dataset.column_names else dataset.column_names[0]

        # Process dataset
        total_loss = 0.0
        total_tokens = 0
        max_seq_len = config.get("max_seq_len", 2048)

        if max_samples:
            if hasattr(dataset, "select"):
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            else:
                dataset = dataset.take(max_samples)

        with torch.no_grad():
            for i, example in enumerate(tqdm(dataset, desc=f"PPL on {dataset_name}")):
                text = example[text_column]
                if not text or len(text.strip()) == 0:
                    continue

                # Tokenize
                encoding = tokenizer(
                    text,
                    max_length=max_seq_len,
                    truncation=True,
                    return_tensors="pt",
                )

                input_ids = encoding["input_ids"].to(device)
                if input_ids.shape[1] < 2:
                    continue

                # Forward pass
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss

                # Accumulate
                num_tokens = input_ids.shape[1] - 1  # Exclude first token
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        # Compute perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss)

        results[dataset_name] = {
            "perplexity": perplexity,
            "avg_loss": avg_loss,
            "total_tokens": total_tokens,
        }

    # Compute average perplexity across datasets
    avg_ppl = sum(r["perplexity"] for r in results.values()) / len(results)
    results["average"] = {"perplexity": avg_ppl}

    return results
