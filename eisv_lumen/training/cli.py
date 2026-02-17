"""Training CLI entry point for EISV-Lumen teacher model pipeline.

Provides subcommands for data preparation, LoRA fine-tuning, evaluation,
and Gate 1 quality checks.

Usage::

    python -m eisv_lumen.training.cli prepare --min-per-shape 50 --output-dir data/training
    python -m eisv_lumen.training.cli train --config configs/teacher.yaml --data-dir data/training
    python -m eisv_lumen.training.cli eval --adapter outputs/teacher_lora/final_adapter --test-data data/training/test.json
    python -m eisv_lumen.training.cli gate1 --results outputs/eval_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from eisv_lumen.training.teacher_train import prepare_training_data
from eisv_lumen.training.teacher_eval import EvalResults, check_gate1


def cmd_prepare(args: argparse.Namespace) -> None:
    """Prepare training data and save to JSON files.

    Calls :func:`prepare_training_data` with synthetic-only data,
    then writes train/val/test splits to the output directory.
    Each example includes ``text``, ``shape``, ``pattern``, and
    ``messages`` (chat message list for tokenizer ``apply_chat_template``).
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Parse shape overrides if provided (format: "shape1:count,shape2:count")
    shape_overrides = None
    if hasattr(args, 'shape_overrides') and args.shape_overrides:
        shape_overrides = {}
        for pair in args.shape_overrides.split(","):
            shape_name, count = pair.strip().split(":")
            shape_overrides[shape_name.strip()] = int(count.strip())
        print(f"  Shape overrides: {shape_overrides}")

    print(f"Preparing training data (min_per_shape={args.min_per_shape}, seed={args.seed})...")
    train, val, test = prepare_training_data(
        real_records=[],
        min_per_shape=args.min_per_shape,
        seed=args.seed,
        shape_overrides=shape_overrides,
    )

    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(output_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  {name}: {len(data)} examples -> {path}")

    print("Done.")


def cmd_train(args: argparse.Namespace) -> None:
    """Run LoRA fine-tuning on the teacher model.

    Loads a :class:`TrainingConfig` (from YAML or defaults) and delegates
    to :func:`train_teacher` which handles model loading, LoRA setup,
    tokenization, and the Hugging Face Trainer loop.

    Requires GPU hardware and ``torch``, ``transformers``, ``peft``
    packages.  Exits with a helpful message if dependencies are missing.
    """
    from eisv_lumen.training.config import load_config
    from eisv_lumen.training.trainer import train_teacher

    config = load_config(args.config)
    adapter_path = train_teacher(config, args.data_dir)
    print(f"Training complete. Adapter saved to: {adapter_path}")


def cmd_eval(args: argparse.Namespace) -> None:
    """Evaluate a fine-tuned teacher model on a test set.

    Loads the LoRA adapter, generates predictions for every test example,
    and computes evaluation metrics.  Results are printed and optionally
    saved to a JSON file.
    """
    from eisv_lumen.training.teacher_inference import evaluate_on_test_set

    results = evaluate_on_test_set(
        adapter_path=args.adapter,
        test_data_path=args.test_data,
        base_model=args.base_model,
    )

    print("=" * 60)
    print("Teacher Evaluation Results")
    print("=" * 60)
    print(f"  Mean Coherence:   {results.mean_coherence:.4f}")
    print(f"  Valid Rate:       {results.valid_rate:.4f}")
    print(f"  Pattern Accuracy: {results.pattern_accuracy:.4f}")
    print(f"  Total:            {results.n_total}")
    print(f"  Valid:            {results.n_valid}")
    print(f"  Diversity:        {results.diversity:.4f}")
    print("=" * 60)

    # Save to JSON
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    results_dict = {
        "mean_coherence": results.mean_coherence,
        "valid_rate": results.valid_rate,
        "pattern_accuracy": results.pattern_accuracy,
        "n_total": results.n_total,
        "n_valid": results.n_valid,
        "per_shape_coherence": results.per_shape_coherence,
        "diversity": results.diversity,
    }
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results saved to: {output_path}")


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
    prep.add_argument(
        "--shape-overrides",
        type=str,
        default=None,
        help="Per-shape min counts, e.g. 'drift_dissonance:800,basin_transition_up:800'",
    )
    prep.set_defaults(func=cmd_prepare)

    # --- train ---
    train = subparsers.add_parser("train", help="Fine-tune teacher model with LoRA")
    train.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML training config file",
    )
    train.add_argument(
        "--data-dir",
        type=str,
        default="data/training",
        help="Directory with train.json and val.json (default: data/training)",
    )
    train.set_defaults(func=cmd_train)

    # --- eval ---
    eval_cmd = subparsers.add_parser("eval", help="Evaluate fine-tuned teacher model")
    eval_cmd.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to saved LoRA adapter directory",
    )
    eval_cmd.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test.json file",
    )
    eval_cmd.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Base model name (default: Qwen/Qwen3-4B)",
    )
    eval_cmd.add_argument(
        "--output",
        type=str,
        default="outputs/eval_results.json",
        help="Path to save evaluation results JSON (default: outputs/eval_results.json)",
    )
    eval_cmd.set_defaults(func=cmd_eval)

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
