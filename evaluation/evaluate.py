#!/usr/bin/env python3
"""
Comprehensive evaluation script for GSA models.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from gsa import GSAForCausalLM
from evaluation.benchmarks import (
    evaluate_perplexity,
    evaluate_mmlu,
    evaluate_gsm8k,
    evaluate_hellaswag,
    evaluate_ruler,
    evaluate_needle_in_haystack,
    analyze_attention_sinks,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GSA model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--config", type=str, default="evaluation/configs/eval_standard.yaml")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=["all"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    if Path(args.config).exists():
        config = OmegaConf.load(args.config)
    else:
        config = OmegaConf.create({})

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = GSAForCausalLM.from_pretrained(args.model_path)
    model = model.to(args.device).eval()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    benchmarks = args.benchmarks if "all" not in args.benchmarks else [
        "perplexity", "mmlu", "gsm8k", "hellaswag",
        "ruler", "needle_in_haystack", "attention_analysis"
    ]

    # Run evaluations
    for benchmark in benchmarks:
        logger.info(f"Running {benchmark} evaluation...")

        try:
            if benchmark == "perplexity":
                results["perplexity"] = evaluate_perplexity(
                    model,
                    config.get("perplexity", {}),
                    args.batch_size
                )
            elif benchmark == "mmlu":
                results["mmlu"] = evaluate_mmlu(
                    model,
                    config.get("mmlu", {}),
                    args.batch_size,
                    args.max_samples
                )
            elif benchmark == "gsm8k":
                results["gsm8k"] = evaluate_gsm8k(
                    model,
                    config.get("gsm8k", {}),
                    args.batch_size,
                    args.max_samples
                )
            elif benchmark == "hellaswag":
                results["hellaswag"] = evaluate_hellaswag(
                    model,
                    config.get("hellaswag", {}),
                    args.batch_size,
                    args.max_samples
                )
            elif benchmark == "ruler":
                results["ruler"] = evaluate_ruler(
                    model,
                    config.get("ruler", {}),
                    args.batch_size
                )
            elif benchmark == "needle_in_haystack":
                results["needle_in_haystack"] = evaluate_needle_in_haystack(
                    model,
                    config.get("needle", {}),
                    args.batch_size
                )
            elif benchmark == "attention_analysis":
                results["attention_analysis"] = analyze_attention_sinks(
                    model,
                    config.get("attention_analysis", {})
                )
        except Exception as e:
            logger.error(f"Error running {benchmark}: {e}")
            results[benchmark] = {"error": str(e)}

    # Save results
    model_name = Path(args.model_path).name
    results_path = output_dir / f"results_{model_name}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    for benchmark, result in results.items():
        if isinstance(result, dict) and "score" in result:
            print(f"{benchmark:30s}: {result['score']:.2f}")
        elif isinstance(result, dict) and "error" in result:
            print(f"{benchmark:30s}: ERROR - {result['error'][:50]}")
        elif isinstance(result, (int, float)):
            print(f"{benchmark:30s}: {result:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
