"""
Long context evaluation (RULER, Needle-in-Haystack).
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from tqdm import tqdm
import random
import string


def evaluate_ruler(
    model: torch.nn.Module,
    config: Dict[str, Any],
    batch_size: int = 1,
) -> Dict[str, Any]:
    """
    Evaluate on RULER benchmark for long context.

    RULER tests various retrieval and reasoning tasks at different context lengths.

    Args:
        model: Model to evaluate
        config: Evaluation configuration
        batch_size: Batch size

    Returns:
        Dictionary with scores at different context lengths
    """
    from transformers import AutoTokenizer

    model.eval()
    device = next(model.parameters()).device

    tokenizer_name = config.get("tokenizer", "meta-llama/Llama-3.1-8B")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Context lengths to test
    context_lengths = config.get("context_lengths", [4096, 8192, 16384, 32768])
    num_samples = config.get("num_samples", 50)

    results = {}

    for ctx_len in context_lengths:
        print(f"Testing RULER at context length {ctx_len}...")

        correct = 0
        total = 0

        for _ in tqdm(range(num_samples), desc=f"RULER @ {ctx_len}"):
            # Generate a retrieval task
            task = _generate_ruler_task(ctx_len, tokenizer)

            # Tokenize
            encoding = tokenizer(task["prompt"], return_tensors="pt", truncation=False)
            input_ids = encoding["input_ids"].to(device)

            # Check if we exceed context limit
            if input_ids.shape[1] > ctx_len:
                input_ids = input_ids[:, :ctx_len]

            with torch.no_grad():
                # Generate answer
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=50,
                    temperature=0.0,
                    do_sample=False,
                )

            response = tokenizer.decode(
                output_ids[0, input_ids.shape[1]:],
                skip_special_tokens=True
            )

            # Check if answer is correct
            if task["answer"].lower() in response.lower():
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        results[ctx_len] = {
            "accuracy": accuracy,
            "score": accuracy * 100,
            "correct": correct,
            "total": total,
        }

    # Average across lengths
    avg_score = sum(r["score"] for r in results.values()) / len(results)
    results["average"] = {"score": avg_score}

    return results


def _generate_ruler_task(context_length: int, tokenizer) -> Dict[str, str]:
    """Generate a single RULER task."""
    # Simple needle-in-haystack style task
    target_key = ''.join(random.choices(string.ascii_lowercase, k=6))
    target_value = ''.join(random.choices(string.ascii_lowercase, k=8))

    # Create distractor sentences
    distractors = []
    num_distractors = context_length // 50  # Roughly 50 tokens per distractor

    for i in range(num_distractors):
        key = ''.join(random.choices(string.ascii_lowercase, k=6))
        value = ''.join(random.choices(string.ascii_lowercase, k=8))
        if key != target_key:
            distractors.append(f"The key {key} has value {value}.")

    # Insert target at random position
    position = random.randint(0, len(distractors))
    distractors.insert(position, f"The key {target_key} has value {target_value}.")

    context = " ".join(distractors)
    prompt = f"{context}\n\nWhat is the value of key {target_key}? Answer:"

    return {
        "prompt": prompt,
        "answer": target_value,
        "position": position / len(distractors),  # Relative position
    }


def evaluate_needle_in_haystack(
    model: torch.nn.Module,
    config: Dict[str, Any],
    batch_size: int = 1,
) -> Dict[str, Any]:
    """
    Evaluate on Needle-in-Haystack benchmark.

    Tests the model's ability to retrieve specific information from long contexts.

    Args:
        model: Model to evaluate
        config: Evaluation configuration
        batch_size: Batch size

    Returns:
        Dictionary with retrieval accuracy at different depths and lengths
    """
    from transformers import AutoTokenizer

    model.eval()
    device = next(model.parameters()).device

    tokenizer_name = config.get("tokenizer", "meta-llama/Llama-3.1-8B")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test configurations
    context_lengths = config.get("context_lengths", [4096, 8192, 16384])
    depths = config.get("depths", [0.0, 0.25, 0.5, 0.75, 1.0])  # Relative depth
    num_samples = config.get("num_samples", 10)

    # The needle to find
    needle_template = "The special magic number is {number}."
    haystack_text = config.get(
        "haystack_text",
        "This is some filler text that serves as the haystack. " * 100
    )

    results = {}

    for ctx_len in context_lengths:
        results[ctx_len] = {}

        for depth in depths:
            print(f"Testing Needle @ length={ctx_len}, depth={depth}...")

            correct = 0
            total = 0

            for _ in range(num_samples):
                # Generate random number for needle
                magic_number = random.randint(1000, 9999)
                needle = needle_template.format(number=magic_number)

                # Create context with needle at specified depth
                haystack_tokens = tokenizer.encode(haystack_text, add_special_tokens=False)

                # Target length (accounting for needle and question)
                target_haystack_len = ctx_len - 200  # Leave room for needle and question

                # Repeat haystack to reach target length
                while len(haystack_tokens) < target_haystack_len:
                    haystack_tokens = haystack_tokens + haystack_tokens

                haystack_tokens = haystack_tokens[:target_haystack_len]

                # Insert needle at depth
                insert_pos = int(len(haystack_tokens) * depth)
                needle_tokens = tokenizer.encode(needle, add_special_tokens=False)

                final_tokens = (
                    haystack_tokens[:insert_pos] +
                    needle_tokens +
                    haystack_tokens[insert_pos:]
                )

                # Create prompt
                context = tokenizer.decode(final_tokens)
                prompt = f"{context}\n\nQuestion: What is the special magic number?\nAnswer:"

                # Tokenize
                encoding = tokenizer(prompt, return_tensors="pt", truncation=False)
                input_ids = encoding["input_ids"].to(device)

                if input_ids.shape[1] > ctx_len:
                    continue  # Skip if too long

                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=20,
                        temperature=0.0,
                        do_sample=False,
                    )

                response = tokenizer.decode(
                    output_ids[0, input_ids.shape[1]:],
                    skip_special_tokens=True
                )

                # Check if answer contains the magic number
                if str(magic_number) in response:
                    correct += 1
                total += 1

            accuracy = correct / total if total > 0 else 0
            results[ctx_len][depth] = {
                "accuracy": accuracy,
                "score": accuracy * 100,
                "correct": correct,
                "total": total,
            }

    # Compute overall score
    all_scores = []
    for ctx_len in context_lengths:
        for depth in depths:
            all_scores.append(results[ctx_len][depth]["score"])

    results["average"] = {"score": sum(all_scores) / len(all_scores)}

    return results
