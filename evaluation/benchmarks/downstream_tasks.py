"""
Downstream task evaluation (MMLU, GSM8K, HellaSwag).
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from tqdm import tqdm
import re


def evaluate_mmlu(
    model: torch.nn.Module,
    config: Dict[str, Any],
    batch_size: int = 8,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate on MMLU benchmark.

    Args:
        model: Model to evaluate
        config: Evaluation configuration
        batch_size: Batch size
        max_samples: Maximum samples

    Returns:
        Dictionary with accuracy scores
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    model.eval()
    device = next(model.parameters()).device

    tokenizer_name = config.get("tokenizer", "meta-llama/Llama-3.1-8B")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load MMLU
    dataset = load_dataset("cais/mmlu", "all", split="test")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    total = 0
    subject_scores = {}

    choices = ["A", "B", "C", "D"]

    with torch.no_grad():
        for example in tqdm(dataset, desc="MMLU"):
            question = example["question"]
            subject = example["subject"]
            answer_idx = example["answer"]

            # Format prompt
            prompt = f"Question: {question}\n"
            for i, choice in enumerate(example["choices"]):
                prompt += f"{choices[i]}. {choice}\n"
            prompt += "Answer:"

            # Tokenize
            encoding = tokenizer(prompt, return_tensors="pt")
            input_ids = encoding["input_ids"].to(device)

            # Get logits for next token
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0, -1, :]

            # Get probabilities for A, B, C, D tokens
            choice_logits = []
            for choice in choices:
                token_id = tokenizer.encode(choice, add_special_tokens=False)[0]
                choice_logits.append(logits[token_id].item())

            # Predict
            predicted = torch.tensor(choice_logits).argmax().item()

            if predicted == answer_idx:
                correct += 1

            # Track per-subject scores
            if subject not in subject_scores:
                subject_scores[subject] = {"correct": 0, "total": 0}
            subject_scores[subject]["total"] += 1
            if predicted == answer_idx:
                subject_scores[subject]["correct"] += 1

            total += 1

    # Compute accuracy
    accuracy = correct / total if total > 0 else 0

    # Compute per-subject accuracy
    for subject in subject_scores:
        s = subject_scores[subject]
        s["accuracy"] = s["correct"] / s["total"] if s["total"] > 0 else 0

    return {
        "score": accuracy * 100,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "subject_scores": subject_scores,
    }


def evaluate_gsm8k(
    model: torch.nn.Module,
    config: Dict[str, Any],
    batch_size: int = 8,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate on GSM8K math reasoning benchmark.

    Args:
        model: Model to evaluate
        config: Evaluation configuration
        batch_size: Batch size
        max_samples: Maximum samples

    Returns:
        Dictionary with accuracy scores
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    model.eval()
    device = next(model.parameters()).device

    tokenizer_name = config.get("tokenizer", "meta-llama/Llama-3.1-8B")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load GSM8K
    dataset = load_dataset("gsm8k", "main", split="test")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    total = 0

    with torch.no_grad():
        for example in tqdm(dataset, desc="GSM8K"):
            question = example["question"]
            answer = example["answer"]

            # Extract numerical answer
            answer_match = re.search(r"####\s*(\d+)", answer)
            if not answer_match:
                continue
            target_answer = int(answer_match.group(1))

            # Format prompt (chain-of-thought style)
            prompt = f"Question: {question}\nLet's solve this step by step.\n"

            # Tokenize
            encoding = tokenizer(prompt, return_tensors="pt")
            input_ids = encoding["input_ids"].to(device)

            # Generate response
            output_ids = model.generate(
                input_ids,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
            )

            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Extract answer from response
            # Look for patterns like "answer is X" or "= X" or just the last number
            numbers = re.findall(r'\d+', response)
            if numbers:
                predicted_answer = int(numbers[-1])
                if predicted_answer == target_answer:
                    correct += 1

            total += 1

    accuracy = correct / total if total > 0 else 0

    return {
        "score": accuracy * 100,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


def evaluate_hellaswag(
    model: torch.nn.Module,
    config: Dict[str, Any],
    batch_size: int = 8,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate on HellaSwag common sense reasoning benchmark.

    Args:
        model: Model to evaluate
        config: Evaluation configuration
        batch_size: Batch size
        max_samples: Maximum samples

    Returns:
        Dictionary with accuracy scores
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    model.eval()
    device = next(model.parameters()).device

    tokenizer_name = config.get("tokenizer", "meta-llama/Llama-3.1-8B")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load HellaSwag
    dataset = load_dataset("hellaswag", split="validation")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    total = 0

    with torch.no_grad():
        for example in tqdm(dataset, desc="HellaSwag"):
            context = example["ctx"]
            endings = example["endings"]
            label = int(example["label"])

            # Compute loss for each ending
            losses = []
            for ending in endings:
                text = context + " " + ending

                encoding = tokenizer(text, return_tensors="pt")
                input_ids = encoding["input_ids"].to(device)

                outputs = model(input_ids=input_ids, labels=input_ids)
                losses.append(outputs.loss.item())

            # Predict the ending with lowest loss
            predicted = torch.tensor(losses).argmin().item()

            if predicted == label:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0

    return {
        "score": accuracy * 100,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }
