#!/usr/bin/env python3
"""Evaluate teacher model on real Lumen trajectories from HuggingFace.

Loads the hikewa/unitares-eisv-trajectories dataset, classifies trajectories
into shapes, generates expression labels using the rule-based system, and
evaluates the V5 teacher model against real data.

Usage:
    python scripts/eval_on_real_data.py \
        --adapter outputs/teacher_lora_v5/final_adapter \
        --output outputs/eval_results_v5_real.json \
        --max-examples 500
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eisv_lumen.extract.derivatives import compute_trajectory_window
from eisv_lumen.shapes.shape_classes import classify_trajectory, TrajectoryShape
from eisv_lumen.training.data_prep import build_training_example
from eisv_lumen.training.chat_format import format_for_tokenizer


def load_hf_dataset(max_examples: int = 0, seed: int = 42) -> List[Dict[str, Any]]:
    """Load real Lumen trajectories from HuggingFace."""
    from datasets import load_dataset

    print("Loading HuggingFace dataset: hikewa/unitares-eisv-trajectories ...")
    ds = load_dataset("hikewa/unitares-eisv-trajectories", split="train")
    print(f"  Total records: {len(ds)}")

    # Filter to real (non-synthetic) data only
    records = []
    for example in ds:
        if example.get("provenance") == "synthetic":
            continue
        records.append(example)

    print(f"  Real (non-synthetic) records: {len(records)}")

    if max_examples > 0 and len(records) > max_examples:
        rng = random.Random(seed)
        records = rng.sample(records, max_examples)
        print(f"  Sampled {max_examples} for evaluation")

    return records


def parse_hf_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a HuggingFace record into states for trajectory classification."""
    states = json.loads(record["eisv_states"])

    # Ensure states have required keys and are valid
    valid_states = []
    for s in states:
        if all(k in s for k in ("t", "E", "I", "S", "V")):
            valid_states.append(s)

    return {
        "states": valid_states,
        "hf_shape": record.get("shape", "unknown"),
        "provenance": record.get("provenance", "unknown"),
    }


def build_real_test_set(
    records: List[Dict[str, Any]],
    min_states: int = 4,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Convert HF records into test examples compatible with eval CLI.

    For each record:
    1. Parse EISV states
    2. Compute trajectory window (derivatives, second derivatives)
    3. Classify trajectory shape
    4. Generate expression using rule-based system
    5. Format for tokenizer

    Returns list of formatted test examples.
    """
    test_examples = []
    shape_counts = Counter()
    skipped = 0
    shape_mismatches = 0
    example_seed = seed

    for i, record in enumerate(records):
        parsed = parse_hf_record(record)
        states = parsed["states"]

        # Need at least min_states for meaningful trajectories
        if len(states) < min_states:
            skipped += 1
            continue

        # Take last min_states for a consistent window size
        window_states = states[-min_states:]

        try:
            # Compute derivatives
            window = compute_trajectory_window(window_states)

            # Classify shape from actual trajectory data
            shape = classify_trajectory(window)
            shape_name = shape.value

            # Track mismatches between HF label and our classification
            if parsed["hf_shape"] != shape_name:
                shape_mismatches += 1

            # Build training example (generates expression via rule-based system)
            training_ex = build_training_example(
                shape=shape_name,
                window=window,
                seed=example_seed,
            )

            # Format for tokenizer (adds messages, text, etc.)
            formatted = format_for_tokenizer(training_ex)
            test_examples.append(formatted)
            shape_counts[shape_name] += 1
            example_seed += 1

        except Exception as e:
            skipped += 1
            if i < 5:  # Only print first few errors
                print(f"  Warning: Skipped record {i}: {e}")
            continue

    print(f"\n  Built {len(test_examples)} test examples from real data")
    print(f"  Skipped: {skipped}")
    print(f"  Shape mismatches (HF label vs classifier): {shape_mismatches}")
    print(f"\n  Shape distribution:")
    for shape, count in sorted(shape_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(test_examples) if test_examples else 0
        print(f"    {shape}: {count} ({pct:.1f}%)")

    return test_examples


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate teacher model on real Lumen trajectories"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="outputs/teacher_lora_v5/final_adapter",
        help="Path to LoRA adapter (default: V5)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Base model name",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/eval_results_v5_real.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=500,
        help="Max examples to evaluate (0 = all, default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    # 1. Load real data from HuggingFace
    records = load_hf_dataset(max_examples=args.max_examples, seed=args.seed)

    # 2. Build test set from real trajectories
    test_examples = build_real_test_set(records, seed=args.seed)

    if not test_examples:
        print("ERROR: No valid test examples generated. Check data format.")
        sys.exit(1)

    # 3. Save test set
    test_path = "data/real_test/test.json"
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    with open(test_path, "w") as f:
        json.dump(test_examples, f, indent=2)
    print(f"\n  Saved real test set to {test_path}")

    # 4. Run evaluation
    print(f"\nEvaluating {args.adapter} on {len(test_examples)} real examples ...")
    from eisv_lumen.training.teacher_inference import evaluate_on_test_set

    results = evaluate_on_test_set(
        adapter_path=args.adapter,
        test_data_path=test_path,
        base_model=args.base_model,
    )

    # 5. Print results
    print("\n" + "=" * 60)
    print("Real Data Evaluation Results")
    print("=" * 60)
    print(f"  Mean Coherence:   {results.mean_coherence:.4f}")
    print(f"  Valid Rate:       {results.valid_rate:.4f}")
    print(f"  Pattern Accuracy: {results.pattern_accuracy:.4f}")
    print(f"  Total:            {results.n_total}")
    print(f"  Valid:            {results.n_valid}")
    print(f"  Diversity:        {results.diversity:.4f}")
    print("-" * 60)
    print("  Per-shape coherence:")
    for shape, coh in sorted(results.per_shape_coherence.items(), key=lambda x: -x[1]):
        print(f"    {shape}: {coh:.4f}")
    print("=" * 60)

    # 6. Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results_dict = {
        "mean_coherence": results.mean_coherence,
        "valid_rate": results.valid_rate,
        "pattern_accuracy": results.pattern_accuracy,
        "n_total": results.n_total,
        "n_valid": results.n_valid,
        "per_shape_coherence": results.per_shape_coherence,
        "diversity": results.diversity,
        "data_source": "real_lumen_trajectories",
        "dataset": "hikewa/unitares-eisv-trajectories",
        "adapter": args.adapter,
    }
    with open(args.output, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
