"""Training CLI entry point for EISV-Lumen teacher model pipeline.

Provides subcommands for data preparation, training (placeholder),
and Gate 1 evaluation.

Usage::

    python -m eisv_lumen.training.cli prepare --min-per-shape 50 --output-dir data/training
    python -m eisv_lumen.training.cli train --config configs/teacher.yaml
    python -m eisv_lumen.training.cli gate1 --results outputs/eval_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from eisv_lumen.training.teacher_train import prepare_training_data
from eisv_lumen.training.teacher_eval import EvalResults, check_gate1


def cmd_prepare(args: argparse.Namespace) -> None:
    """Prepare training data and save to JSON files.

    Calls :func:`prepare_training_data` with synthetic-only data,
    then writes train/val/test splits to the output directory.
    The ``messages`` key is stripped before serialization since it
    contains non-trivially-serializable chat message objects.
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Preparing training data (min_per_shape={args.min_per_shape}, seed={args.seed})...")
    train, val, test = prepare_training_data(
        real_records=[],
        min_per_shape=args.min_per_shape,
        seed=args.seed,
    )

    def _strip_messages(items: List[dict]) -> List[dict]:
        """Remove 'messages' key from each dict for JSON serialization."""
        return [{k: v for k, v in item.items() if k != "messages"} for item in items]

    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(output_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(_strip_messages(data), f, indent=2)
        print(f"  {name}: {len(data)} examples -> {path}")

    print("Done.")


def cmd_train(args: argparse.Namespace) -> None:
    """Placeholder for GPU-based teacher model training.

    Actual LoRA fine-tuning requires GPU hardware and is not yet
    implemented in this CLI. See the training configuration module
    for the planned approach.
    """
    config_path = args.config
    print(f"GPU training not yet implemented.")
    print(f"Config would be loaded from: {config_path or '(default)'}")
    print("Use the training notebook or a dedicated GPU script for actual training.")


def cmd_gate1(args: argparse.Namespace) -> None:
    """Load evaluation results and run Gate 1 check.

    Loads an :class:`EvalResults` from a JSON file and checks whether
    the teacher model passes the Gate 1 quality thresholds.
    Exits with code 0 if passed, 1 if failed.
    """
    results_path = args.results
    with open(results_path, "r") as f:
        data = json.load(f)

    results = EvalResults(
        mean_coherence=data["mean_coherence"],
        valid_rate=data["valid_rate"],
        pattern_accuracy=data.get("pattern_accuracy", 0.0),
        n_total=data.get("n_total", 0),
        n_valid=data.get("n_valid", 0),
        per_shape_coherence=data.get("per_shape_coherence", {}),
        diversity=data.get("diversity", 0.0),
    )

    passed, reasons = check_gate1(results)

    print("=" * 60)
    print("Gate 1 Evaluation Results")
    print("=" * 60)
    print(f"  Mean Coherence:   {results.mean_coherence:.4f}")
    print(f"  Valid Rate:       {results.valid_rate:.4f}")
    print(f"  Pattern Accuracy: {results.pattern_accuracy:.4f}")
    print(f"  Total:            {results.n_total}")
    print(f"  Valid:            {results.n_valid}")
    print(f"  Diversity:        {results.diversity:.4f}")
    print("-" * 60)

    if passed:
        print("GATE 1: PASSED")
    else:
        print("GATE 1: FAILED")
        for reason in reasons:
            print(f"  - {reason}")

    print("=" * 60)
    sys.exit(0 if passed else 1)


def main() -> None:
    """Main CLI entry point with argparse subcommands."""
    parser = argparse.ArgumentParser(
        prog="eisv-lumen-train",
        description="EISV-Lumen teacher model training pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- prepare ---
    prep = subparsers.add_parser("prepare", help="Prepare training data")
    prep.add_argument(
        "--min-per-shape",
        type=int,
        default=50,
        help="Minimum examples per shape class (default: 50)",
    )
    prep.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    prep.add_argument(
        "--output-dir",
        type=str,
        default="data/training",
        help="Output directory for JSON files (default: data/training)",
    )
    prep.set_defaults(func=cmd_prepare)

    # --- train ---
    train = subparsers.add_parser("train", help="Train teacher model (placeholder)")
    train.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML training config file",
    )
    train.set_defaults(func=cmd_train)

    # --- gate1 ---
    gate1 = subparsers.add_parser("gate1", help="Run Gate 1 quality check")
    gate1.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to evaluation results JSON file",
    )
    gate1.set_defaults(func=cmd_gate1)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
