#!/usr/bin/env python3
"""Generate distillation dataset by running V6 teacher on diverse inputs.

For each of 9 trajectory shapes, generates diverse input windows (from real
HF data + synthetic), runs the teacher model, and collects structured outputs.

Usage:
    python scripts/generate_distillation_data.py \
        --adapter outputs/teacher_lora_v6/final_adapter \
        --output data/distillation/teacher_outputs.json \
        --per-shape 600
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eisv_lumen.training.data_prep import format_trajectory_input
from eisv_lumen.training.teacher_train import parse_model_output
from eisv_lumen.shapes.shape_classes import TrajectoryShape
from eisv_lumen.synthetic.trajectory_generator import generate_trajectory


def load_real_windows_from_hf(
    max_per_shape: int = 300,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load real trajectory windows from HuggingFace dataset, grouped by shape.

    Uses our classifier to assign shapes (may differ from HF labels).
    """
    from datasets import load_dataset
    from eisv_lumen.shapes.shape_classes import classify_trajectory

    print("Loading HuggingFace dataset ...")
    ds = load_dataset("hikewa/unitares-eisv-trajectories", split="train")

    rng = random.Random(seed)
    by_shape: Dict[str, List[Dict[str, Any]]] = {s.value: [] for s in TrajectoryShape}

    indices = list(range(len(ds)))
    rng.shuffle(indices)

    for idx in indices:
        row = ds[idx]
        states = json.loads(row["eisv_states"])
        if len(states) < 4:
            continue

        # Use last 20 states (or all if fewer)
        window_states = states[-20:]
        n = len(window_states)

        # Compute derivatives
        derivs = []
        for i in range(1, n):
            dt = window_states[i]["t"] - window_states[i - 1]["t"]
            if dt <= 0:
                dt = 1.0
            derivs.append({
                "dE": (window_states[i]["E"] - window_states[i - 1]["E"]) / dt,
                "dI": (window_states[i]["I"] - window_states[i - 1]["I"]) / dt,
                "dS": (window_states[i]["S"] - window_states[i - 1]["S"]) / dt,
                "dV": (window_states[i]["V"] - window_states[i - 1]["V"]) / dt,
            })

        # Second derivatives
        second_derivs = []
        for i in range(1, len(derivs)):
            dt = window_states[i + 1]["t"] - window_states[i]["t"]
            if dt <= 0:
                dt = 1.0
            second_derivs.append({
                "d2E": (derivs[i]["dE"] - derivs[i - 1]["dE"]) / dt,
                "d2I": (derivs[i]["dI"] - derivs[i - 1]["dI"]) / dt,
                "d2S": (derivs[i]["dS"] - derivs[i - 1]["dS"]) / dt,
                "d2V": (derivs[i]["dV"] - derivs[i - 1]["dV"]) / dt,
            })

        # Build window dict
        window = {
            "states": window_states,
            "derivatives": derivs,
            "second_derivatives": second_derivs,
        }

        # Classify
        try:
            shape = classify_trajectory(window)
        except (KeyError, IndexError, ZeroDivisionError):
            continue
        if shape is None:
            continue

        shape_name = shape.value
        if len(by_shape[shape_name]) >= max_per_shape:
            continue

        by_shape[shape_name].append(window)

        # Check if all shapes have enough
        if all(len(v) >= max_per_shape for v in by_shape.values()):
            break

    for shape, windows in by_shape.items():
        print(f"  {shape}: {len(windows)} real windows")

    return by_shape


def generate_synthetic_windows(
    shape: str,
    count: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate synthetic trajectory windows for a given shape."""
    rng = random.Random(seed)
    windows = []

    for i in range(count):
        try:
            states = generate_trajectory(
                shape,
                n_points=20,
                seed=rng.randint(0, 2**31),
            )

            # Compute derivatives
            n = len(states)
            derivs = []
            for j in range(1, n):
                dt = states[j]["t"] - states[j - 1]["t"]
                if dt <= 0:
                    dt = 1.0
                derivs.append({
                    "dE": (states[j]["E"] - states[j - 1]["E"]) / dt,
                    "dI": (states[j]["I"] - states[j - 1]["I"]) / dt,
                    "dS": (states[j]["S"] - states[j - 1]["S"]) / dt,
                    "dV": (states[j]["V"] - states[j - 1]["V"]) / dt,
                })

            # Second derivatives
            second_derivs = []
            for j in range(1, len(derivs)):
                dt = states[j + 1]["t"] - states[j]["t"]
                if dt <= 0:
                    dt = 1.0
                second_derivs.append({
                    "d2E": (derivs[j]["dE"] - derivs[j - 1]["dE"]) / dt,
                    "d2I": (derivs[j]["dI"] - derivs[j - 1]["dI"]) / dt,
                    "d2S": (derivs[j]["dS"] - derivs[j - 1]["dS"]) / dt,
                    "d2V": (derivs[j]["dV"] - derivs[j - 1]["dV"]) / dt,
                })

            windows.append({
                "states": states,
                "derivatives": derivs,
                "second_derivatives": second_derivs,
            })
        except (ValueError, RuntimeError):
            continue

    return windows


def extract_features(shape: str, window: Dict[str, Any]) -> Dict[str, float]:
    """Extract numeric features from a trajectory window."""
    states = window["states"]
    derivs = window.get("derivatives", [])
    second = window.get("second_derivatives", [])

    def mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "shape": shape,
        "mean_E": mean([s["E"] for s in states]),
        "mean_I": mean([s["I"] for s in states]),
        "mean_S": mean([s["S"] for s in states]),
        "mean_V": mean([s["V"] for s in states]),
        "dE": mean([d["dE"] for d in derivs]) if derivs else 0.0,
        "dI": mean([d["dI"] for d in derivs]) if derivs else 0.0,
        "dS": mean([d["dS"] for d in derivs]) if derivs else 0.0,
        "dV": mean([d["dV"] for d in derivs]) if derivs else 0.0,
        "d2E": mean([d["d2E"] for d in second]) if second else 0.0,
        "d2I": mean([d["d2I"] for d in second]) if second else 0.0,
        "d2S": mean([d["d2S"] for d in second]) if second else 0.0,
        "d2V": mean([d["d2V"] for d in second]) if second else 0.0,
    }


def run_teacher_on_windows(
    adapter_path: str,
    shape_windows: Dict[str, List[Dict[str, Any]]],
    base_model: str = "Qwen/Qwen3-4B",
) -> List[Dict[str, Any]]:
    """Run teacher model on all windows, collect outputs."""
    from eisv_lumen.training.teacher_inference import (
        load_teacher_model,
        generate_expression,
    )

    print(f"\nLoading teacher model from {adapter_path} ...")
    model, tokenizer = load_teacher_model(adapter_path, base_model)

    results = []
    total = sum(len(ws) for ws in shape_windows.values())
    done = 0
    skipped = 0

    for shape, windows in shape_windows.items():
        for window in windows:
            # Format input for teacher
            trajectory_input = format_trajectory_input(shape, window)

            # Generate
            raw_output = generate_expression(model, tokenizer, trajectory_input)

            # Parse
            parsed = parse_model_output(raw_output)

            if not parsed.valid:
                skipped += 1
                done += 1
                continue

            # Extract features + teacher output
            features = extract_features(shape, window)
            features["token_1"] = parsed.eisv_tokens[0] if parsed.eisv_tokens else ""
            features["token_2"] = parsed.eisv_tokens[1] if len(parsed.eisv_tokens) > 1 else "none"
            features["token_3"] = parsed.eisv_tokens[2] if len(parsed.eisv_tokens) > 2 else "none"
            features["pattern"] = parsed.pattern

            results.append(features)

            done += 1
            if done % 50 == 0:
                print(f"  [{done}/{total}] processed ({skipped} skipped) ...")

    print(f"\nTotal: {len(results)} valid outputs, {skipped} skipped (invalid parse)")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate distillation dataset from V6 teacher"
    )
    parser.add_argument(
        "--adapter", required=True,
        help="Path to V6 teacher adapter directory",
    )
    parser.add_argument(
        "--output", default="data/distillation/teacher_outputs.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--per-shape", type=int, default=600,
        help="Target examples per shape (default: 600)",
    )
    parser.add_argument(
        "--base-model", default="Qwen/Qwen3-4B",
        help="Base model name",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load real data from HuggingFace
    real_windows = load_real_windows_from_hf(
        max_per_shape=args.per_shape,
        seed=args.seed,
    )

    # Backfill with synthetic for rare shapes
    shape_windows: Dict[str, List[Dict[str, Any]]] = {}
    for shape in TrajectoryShape:
        name = shape.value
        real = real_windows.get(name, [])
        need = max(0, args.per_shape - len(real))
        synthetic = []
        if need > 0:
            print(f"  Backfilling {name}: {need} synthetic windows ...")
            synthetic = generate_synthetic_windows(name, need, seed=args.seed + hash(name))

        combined = real + synthetic
        random.Random(args.seed).shuffle(combined)
        shape_windows[name] = combined[:args.per_shape]
        print(f"  {name}: {len(real)} real + {len(synthetic)} synthetic = {len(shape_windows[name])} total")

    # Run teacher on all windows
    results = run_teacher_on_windows(
        args.adapter, shape_windows, args.base_model
    )

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} examples to {args.output}")

    # Summary
    from collections import Counter
    shape_counts = Counter(r["shape"] for r in results)
    pattern_counts = Counter(r["pattern"] for r in results)
    token1_counts = Counter(r["token_1"] for r in results)

    print("\nShape distribution:")
    for shape, count in shape_counts.most_common():
        print(f"  {shape}: {count}")

    print("\nPattern distribution:")
    for pattern, count in pattern_counts.most_common():
        print(f"  {pattern}: {count}")

    print(f"\nUnique token_1 values: {len(token1_counts)}")
    print(f"Top 5 token_1: {token1_counts.most_common(5)}")


if __name__ == "__main__":
    main()
