#!/usr/bin/env python3
"""Prepare blended real + synthetic training data for V6.

Loads real Lumen trajectories from HuggingFace, classifies them, generates
expression labels via rule-based system, then backfills rare shapes with
synthetic data to ensure all 9 shapes are well-represented.

Strategy:
- Use up to max_real_per_shape real examples per shape
- Backfill any shape below min_per_shape with synthetic data
- Hold out real examples for test set (evaluated separately)

Usage:
    python scripts/prepare_blended_data.py \
        --max-real-per-shape 400 \
        --min-per-shape 400 \
        --output-dir data/training_v6
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eisv_lumen.extract.derivatives import compute_trajectory_window
from eisv_lumen.shapes.shape_classes import classify_trajectory, TrajectoryShape
from eisv_lumen.training.data_prep import build_training_example
from eisv_lumen.training.chat_format import format_for_tokenizer
from eisv_lumen.synthetic.trajectory_generator import generate_trajectory


def load_and_classify_real_data(seed: int = 42, window_size: int = 4) -> Dict[str, List[Dict]]:
    """Load real trajectories from HF and group by classified shape."""
    from datasets import load_dataset

    print("Loading HuggingFace dataset ...")
    ds = load_dataset("hikewa/unitares-eisv-trajectories", split="train")
    print(f"  Total records: {len(ds)}")

    by_shape = defaultdict(list)
    skipped = 0

    for example in ds:
        if example.get("provenance") == "synthetic":
            continue

        states = json.loads(example["eisv_states"])
        if len(states) < window_size:
            skipped += 1
            continue

        # Take last N states for consistent window
        window_states = states[-window_size:]

        try:
            window = compute_trajectory_window(window_states)
            shape = classify_trajectory(window)
            by_shape[shape.value].append(window)
        except Exception:
            skipped += 1
            continue

    print(f"  Classified {sum(len(v) for v in by_shape.values())} real trajectories")
    print(f"  Skipped: {skipped}")
    for shape in sorted(by_shape.keys()):
        print(f"    {shape}: {len(by_shape[shape])}")

    return dict(by_shape)


def build_blended_dataset(
    real_by_shape: Dict[str, List[Dict]],
    max_real_per_shape: int = 400,
    min_per_shape: int = 400,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Build blended dataset: real examples + synthetic backfill."""
    rng = random.Random(seed)
    all_examples = []
    example_seed = seed
    real_count = 0
    synth_count = 0

    for shape_enum in TrajectoryShape:
        shape = shape_enum.value
        real_windows = real_by_shape.get(shape, [])

        # Sample up to max_real_per_shape real examples
        if len(real_windows) > max_real_per_shape:
            sampled = rng.sample(real_windows, max_real_per_shape)
        else:
            sampled = list(real_windows)
        rng.shuffle(sampled)

        # Build examples from real data
        for window in sampled:
            ex = build_training_example(shape, window, seed=example_seed)
            formatted = format_for_tokenizer(ex)
            formatted["provenance"] = "real"
            all_examples.append(formatted)
            example_seed += 1
            real_count += 1

        # Backfill with synthetic if below minimum
        deficit = min_per_shape - len(sampled)
        if deficit > 0:
            synth_seed = seed + 100000 + hash(shape) % 10000
            for i in range(deficit):
                states = generate_trajectory(shape, seed=synth_seed + i)
                window = compute_trajectory_window(states)
                ex = build_training_example(shape, window, seed=synth_seed + i + 50000)
                formatted = format_for_tokenizer(ex)
                formatted["provenance"] = "synthetic"
                all_examples.append(formatted)
                synth_count += 1

        total_for_shape = len(sampled) + max(0, deficit)
        real_for_shape = len(sampled)
        synth_for_shape = max(0, deficit)
        print(f"  {shape}: {total_for_shape} total ({real_for_shape} real, {synth_for_shape} synthetic)")

    print(f"\n  Total: {len(all_examples)} ({real_count} real, {synth_count} synthetic)")
    return all_examples


def split_dataset(examples, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Stratified split by shape."""
    rng = random.Random(seed)
    by_shape = defaultdict(list)
    for ex in examples:
        by_shape[ex["shape"]].append(ex)

    train, val, test = [], [], []
    for shape in sorted(by_shape.keys()):
        group = by_shape[shape]
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio)) if n > 2 else 0
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1) if n > 1 else 0
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Prepare blended real+synthetic data")
    parser.add_argument("--max-real-per-shape", type=int, default=400)
    parser.add_argument("--min-per-shape", type=int, default=400)
    parser.add_argument("--window-size", type=int, default=4,
                        help="Number of EISV states per trajectory window (default: 4, recommended: 15-20)")
    parser.add_argument("--output-dir", type=str, default="data/training_v6")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Load and classify real data
    real_by_shape = load_and_classify_real_data(seed=args.seed, window_size=args.window_size)

    # 2. Build blended dataset
    print(f"\nBuilding blended dataset (max_real={args.max_real_per_shape}, min={args.min_per_shape}) ...")
    examples = build_blended_dataset(
        real_by_shape,
        max_real_per_shape=args.max_real_per_shape,
        min_per_shape=args.min_per_shape,
        seed=args.seed,
    )

    # 3. Split
    train, val, test = split_dataset(examples, seed=args.seed)

    # 4. Save
    os.makedirs(args.output_dir, exist_ok=True)
    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(args.output_dir, f"{name}.json")
        # Strip provenance before saving (not needed for training)
        clean = [{k: v for k, v in ex.items() if k != "provenance"} for ex in data]
        with open(path, "w") as f:
            json.dump(clean, f, indent=2)
        # Count real vs synthetic
        n_real = sum(1 for ex in data if ex.get("provenance") == "real")
        n_synth = sum(1 for ex in data if ex.get("provenance") == "synthetic")
        print(f"  {name}: {len(data)} examples ({n_real} real, {n_synth} synthetic) -> {path}")

    print("Done.")


if __name__ == "__main__":
    main()
