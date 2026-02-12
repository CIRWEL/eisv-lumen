"""Full evaluation pipeline with go/no-go gate for Layer 2 integration.

Runs the complete EISV trajectory analysis on all real Lumen data:
extraction, assembly, baseline evaluation, expression generation
(with and without feedback), and produces a structured JSON report
with go/no-go decision for proceeding to Layer 2.

Usage:
    python3 -m eisv_lumen.scripts.full_evaluation [/path/to/anima.db]
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List

from eisv_lumen.eval.baseline import BaselineCondition, evaluate_baseline
from eisv_lumen.eval.metrics import expression_trajectory_coherence, vocabulary_diversity_per_shape
from eisv_lumen.extract.assembler import assemble_dataset
from eisv_lumen.extract.lumen_expressions import extract_primitive_history
from eisv_lumen.extract.lumen_states import extract_state_history
from eisv_lumen.shapes.expression_generator import ExpressionGenerator
from eisv_lumen.shapes.shape_classes import TrajectoryShape

DEFAULT_DB_PATH = "/Users/cirwel/.anima/anima.db"

ALL_SHAPE_VALUES = [s.value for s in TrajectoryShape]


def compute_shape_distribution(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute shape distribution statistics from trajectory records.

    Returns a dict mapping shape name to {count, percent}.
    """
    total = len(records)
    if total == 0:
        return {}
    counts = Counter(r["shape"] for r in records)
    return {
        shape: {
            "count": counts.get(shape, 0),
            "percent": round(100.0 * counts.get(shape, 0) / total, 2),
        }
        for shape in sorted(ALL_SHAPE_VALUES)
        if counts.get(shape, 0) > 0
    }


def evaluate_expression_generator(
    records: List[Dict[str, Any]],
    seed: int = 42,
) -> Dict[str, Any]:
    """Run ExpressionGenerator on records and compute coherence metrics.

    Returns dict with mean_coherence, per_shape_coherence, and diversity.
    """
    if not records:
        return {
            "mean_coherence": 0.0,
            "per_shape_coherence": {},
            "diversity": {},
        }

    gen = ExpressionGenerator(seed=seed)
    coherence_scores: List[float] = []
    per_shape_scores: Dict[str, List[float]] = defaultdict(list)
    augmented: List[Dict[str, Any]] = []

    for rec in records:
        shape = rec["shape"]
        tokens = gen.generate(shape)
        score = expression_trajectory_coherence(shape, tokens)
        coherence_scores.append(score)
        per_shape_scores[shape].append(score)
        augmented.append({"shape": shape, "tokens": tokens})

    mean_coh = sum(coherence_scores) / len(coherence_scores)
    per_shape_coh = {
        shape: round(sum(scores) / len(scores), 4)
        for shape, scores in per_shape_scores.items()
    }
    diversity = vocabulary_diversity_per_shape(augmented)

    return {
        "mean_coherence": round(mean_coh, 4),
        "per_shape_coherence": per_shape_coh,
        "diversity": {k: round(v, 4) for k, v in diversity.items()},
    }


def evaluate_expression_generator_with_feedback(
    records: List[Dict[str, Any]],
    seed: int = 42,
) -> Dict[str, Any]:
    """Run ExpressionGenerator with online feedback loop.

    For each record: generate -> score -> update_weights, then
    re-generate a second pass to measure improvement.

    Returns dict with mean_coherence and per_shape_coherence.
    """
    if not records:
        return {
            "mean_coherence": 0.0,
            "per_shape_coherence": {},
        }

    gen = ExpressionGenerator(seed=seed)

    # Pass 1: generate, score, and update weights
    for rec in records:
        shape = rec["shape"]
        tokens = gen.generate(shape)
        score = expression_trajectory_coherence(shape, tokens)
        gen.update_weights(shape, tokens, score)

    # Pass 2: re-generate with updated weights and measure coherence
    # Use a fresh RNG for pass 2 to get independent samples,
    # but keep the learned weights.
    gen.rng.seed(seed + 1)

    coherence_scores: List[float] = []
    per_shape_scores: Dict[str, List[float]] = defaultdict(list)

    for rec in records:
        shape = rec["shape"]
        tokens = gen.generate(shape)
        score = expression_trajectory_coherence(shape, tokens)
        coherence_scores.append(score)
        per_shape_scores[shape].append(score)

    mean_coh = sum(coherence_scores) / len(coherence_scores)
    per_shape_coh = {
        shape: round(sum(scores) / len(scores), 4)
        for shape, scores in per_shape_scores.items()
    }

    return {
        "mean_coherence": round(mean_coh, 4),
        "per_shape_coherence": per_shape_coh,
    }


def compute_go_no_go(report: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate go/no-go criteria from a report.

    Criteria:
    - beats_random: expression_generator mean_coherence > random + 0.05
    - min_shapes_observed: at least 3 distinct shapes in data
    - feedback_improves: with-feedback coherence >= without-feedback coherence

    Returns a dict with decision, criteria details, and all_passed flag.
    """
    random_coh = report.get("baselines", {}).get("random", {}).get("mean_coherence", 0.0)
    gen_coh = report.get("expression_generator", {}).get("mean_coherence", 0.0)
    fb_coh = report.get("expression_generator_with_feedback", {}).get("mean_coherence", 0.0)
    shapes_observed = report.get("shapes_observed", 0)

    beats_random_val = round(gen_coh - random_coh, 4)
    beats_random_passed = beats_random_val > 0.05

    min_shapes_passed = shapes_observed >= 3

    feedback_improvement = round(fb_coh - gen_coh, 4)
    feedback_passed = fb_coh >= gen_coh

    all_passed = beats_random_passed and min_shapes_passed and feedback_passed

    return {
        "decision": "GO" if all_passed else "NO-GO",
        "criteria": {
            "beats_random": {
                "passed": beats_random_passed,
                "value": beats_random_val,
                "threshold": 0.05,
            },
            "min_shapes_observed": {
                "passed": min_shapes_passed,
                "value": shapes_observed,
                "threshold": 3,
            },
            "feedback_improves": {
                "passed": feedback_passed,
                "value": feedback_improvement,
                "threshold": 0.0,
            },
        },
        "all_passed": all_passed,
    }


def run_full_evaluation(db_path: str) -> Dict[str, Any]:
    """Run the full evaluation pipeline and return the report dict.

    Steps:
    1. Extract all state_history with EISV computation
    2. Extract all primitive_history
    3. Assemble dataset (window_size=20, stride=10)
    4. Run all 3 baseline conditions
    5. Run ExpressionGenerator (without feedback)
    6. Run ExpressionGenerator with feedback loop
    7. Compute shape distribution and go/no-go gate
    """
    # Step 1-2: Extract data
    print("Extracting state_history...", file=sys.stderr)
    states = extract_state_history(db_path, compute_eisv=True)
    print(f"  -> {len(states)} states", file=sys.stderr)

    print("Extracting primitive_history...", file=sys.stderr)
    expressions = extract_primitive_history(db_path)
    print(f"  -> {len(expressions)} expressions", file=sys.stderr)

    # Step 3: Assemble dataset
    print("Assembling dataset (window=20, stride=10)...", file=sys.stderr)
    records = assemble_dataset(states, expressions, window_size=20, stride=10)
    print(f"  -> {len(records)} trajectory windows", file=sys.stderr)

    # Step 4: Shape distribution
    shape_dist = compute_shape_distribution(records)
    observed_shapes = list(shape_dist.keys())
    missing_shapes = [s for s in ALL_SHAPE_VALUES if s not in observed_shapes]

    # Step 5: Baselines
    print("Running baselines...", file=sys.stderr)
    baselines: Dict[str, Any] = {}
    for condition in BaselineCondition:
        print(f"  -> {condition.value}...", file=sys.stderr)
        result = evaluate_baseline(records, condition, n_tokens=3, seed=42)
        baselines[condition.value] = {
            "mean_coherence": round(result["mean_coherence"], 4),
            "per_shape": {
                k: round(v, 4) for k, v in result["per_shape_coherence"].items()
            },
        }

    # Step 6: Expression generator (no feedback)
    print("Running expression generator...", file=sys.stderr)
    gen_result = evaluate_expression_generator(records, seed=42)

    # Step 7: Expression generator with feedback
    print("Running expression generator with feedback...", file=sys.stderr)
    fb_result = evaluate_expression_generator_with_feedback(records, seed=42)

    # Build report (without go/no-go yet)
    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_summary": {
            "total_states": len(states),
            "total_expressions": len(expressions),
            "total_trajectory_windows": len(records),
            "window_size": 20,
            "stride": 10,
        },
        "shape_distribution": shape_dist,
        "shapes_observed": len(observed_shapes),
        "shapes_missing": missing_shapes,
        "baselines": baselines,
        "expression_generator": gen_result,
        "expression_generator_with_feedback": {
            "mean_coherence": fb_result["mean_coherence"],
            "per_shape_coherence": fb_result["per_shape_coherence"],
            "improvement_over_no_feedback": round(
                fb_result["mean_coherence"] - gen_result["mean_coherence"], 4
            ),
        },
    }

    # Step 8: Go/no-go
    print("Computing go/no-go gate...", file=sys.stderr)
    report["go_no_go"] = compute_go_no_go(report)

    return report


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB_PATH
    report = run_full_evaluation(db_path)
    print(json.dumps(report, indent=2))
