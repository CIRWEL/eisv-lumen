"""Evaluate student classifiers against teacher outputs and rule-based baseline.

Computes coherence (fraction of tokens affine to shape), agreement rate
with teacher, and compares to the rule-based ExpressionGenerator baseline.

Usage:
    python -m eisv_lumen.distillation.eval_student \
        --models outputs/student \
        --data data/distillation/teacher_outputs.json \
        --output outputs/student/eval_results.json
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from eisv_lumen.distillation.train_student import (
    StudentModels,
    load_distillation_data,
    load_student_models,
    predict,
    NUMERIC_FEATURES,
)
from eisv_lumen.eval.metrics import SHAPE_TOKEN_AFFINITY


def compute_coherence(shape: str, tokens: List[str]) -> float:
    """Fraction of tokens that are affine to the given shape."""
    affine = set(SHAPE_TOKEN_AFFINITY.get(shape, []))
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in affine) / len(tokens)


def evaluate_student(
    models: StudentModels,
    test_data: List[Dict[str, Any]],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate student classifiers on held-out test data.

    Returns metrics dict with coherence, agreement rates, per-shape breakdown.
    """
    coherences: List[float] = []
    token1_matches = 0
    pattern_matches = 0
    full_matches = 0
    total = 0

    per_shape: Dict[str, List[float]] = defaultdict(list)

    for row in test_data:
        shape = row["shape"]
        features = {f: row[f] for f in NUMERIC_FEATURES}

        # Student prediction
        pred = predict(models, shape, features)

        # Teacher ground truth
        teacher_tokens = [row["token_1"]]
        if row.get("token_2", "none") != "none":
            teacher_tokens.append(row["token_2"])
        if row.get("token_3", "none") != "none":
            teacher_tokens.append(row["token_3"])

        # Coherence (student tokens vs shape affinity)
        coh = compute_coherence(shape, pred["eisv_tokens"])
        coherences.append(coh)
        per_shape[shape].append(coh)

        # Agreement with teacher
        if pred["token_1"] == row["token_1"]:
            token1_matches += 1
        if pred["pattern"] == row["pattern"]:
            pattern_matches += 1
        if pred["token_1"] == row["token_1"] and pred["pattern"] == row["pattern"]:
            if pred.get("token_2", "none") == row.get("token_2", "none"):
                full_matches += 1

        total += 1

    mean_coherence = float(np.mean(coherences))
    token1_agreement = token1_matches / total if total else 0.0
    pattern_agreement = pattern_matches / total if total else 0.0
    full_agreement = full_matches / total if total else 0.0

    # Per-shape coherence
    shape_coherence = {}
    for shape, vals in sorted(per_shape.items()):
        shape_coherence[shape] = {
            "mean": float(np.mean(vals)),
            "count": len(vals),
        }

    results = {
        "mean_coherence": mean_coherence,
        "token1_agreement": token1_agreement,
        "pattern_agreement": pattern_agreement,
        "full_agreement": full_agreement,
        "total_examples": total,
        "per_shape_coherence": shape_coherence,
    }

    if verbose:
        print("=" * 60)
        print("Student Evaluation Results")
        print("=" * 60)
        print(f"  Mean Coherence:      {mean_coherence:.4f}")
        print(f"  Token-1 Agreement:   {token1_agreement:.4f}")
        print(f"  Pattern Agreement:   {pattern_agreement:.4f}")
        print(f"  Full Agreement:      {full_agreement:.4f}")
        print(f"  Total Examples:      {total}")
        print("-" * 60)
        print("  Per-shape coherence:")
        for shape, info in sorted(shape_coherence.items(), key=lambda x: -x[1]["mean"]):
            print(f"    {shape}: {info['mean']:.4f} (n={info['count']})")
        print("=" * 60)

        # Gate 2 check
        gate2_pass = mean_coherence >= 0.90
        print(f"\n  Gate 2 (coherence ≥ 0.90): {'PASS ✓' if gate2_pass else 'FAIL ✗'}")

    return results


def evaluate_baseline(
    test_data: List[Dict[str, Any]],
    n_samples: int = 10,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate rule-based ExpressionGenerator as baseline.

    Runs each example n_samples times (since baseline is stochastic)
    and reports average coherence.
    """
    from eisv_lumen.shapes.expression_generator import ExpressionGenerator

    gen = ExpressionGenerator(seed=seed)
    coherences: List[float] = []

    for row in test_data:
        shape = row["shape"]
        shape_coherences = []
        for _ in range(n_samples):
            tokens = gen.generate(shape)
            coh = compute_coherence(shape, tokens)
            shape_coherences.append(coh)
        coherences.append(float(np.mean(shape_coherences)))

    mean_coherence = float(np.mean(coherences))

    if verbose:
        print(f"\n  Rule-based baseline coherence: {mean_coherence:.4f}")
        print(f"  (averaged over {n_samples} samples per example)")

    return {"baseline_coherence": mean_coherence}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate student classifiers")
    parser.add_argument(
        "--models", required=True,
        help="Path to student model directory",
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to teacher_outputs.json",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional: save results JSON",
    )
    args = parser.parse_args()

    # Load
    models = load_student_models(args.models)
    data = load_distillation_data(args.data)

    # Use 20% as test set (same split as training)
    from sklearn.model_selection import train_test_split
    _, test_data = train_test_split(data, test_size=0.2, random_state=42)

    print(f"Evaluating on {len(test_data)} held-out examples ...\n")

    # Evaluate student
    student_results = evaluate_student(models, test_data)

    # Evaluate baseline
    baseline_results = evaluate_baseline(test_data)

    # Combine
    results = {**student_results, **baseline_results}
    improvement = student_results["mean_coherence"] - baseline_results["baseline_coherence"]
    results["improvement_over_baseline"] = improvement
    print(f"\n  Student improvement over baseline: {improvement:+.4f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
