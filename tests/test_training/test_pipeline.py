"""Integration tests for the full EISV-Lumen teacher training pipeline.

Tests the end-to-end flow from trajectory generation through training
data preparation to evaluation and gate checking.
"""

from __future__ import annotations

import pytest

from eisv_lumen.shapes.shape_classes import TrajectoryShape, classify_trajectory
from eisv_lumen.synthetic.trajectory_generator import generate_trajectory
from eisv_lumen.extract.derivatives import compute_trajectory_window
from eisv_lumen.training.data_prep import build_training_example
from eisv_lumen.training.chat_format import format_chat_messages
from eisv_lumen.training.teacher_train import (
    prepare_training_data,
    parse_model_output,
    validate_output,
)
from eisv_lumen.training.teacher_eval import (
    evaluate_predictions,
    check_gate1,
)


class TestFullPipeline:
    """Integration tests for the full trajectory-to-evaluation pipeline."""

    def test_trajectory_to_training_example(self):
        """For each shape: generate -> classify -> build example -> format -> verify."""
        for shape in TrajectoryShape:
            # Generate trajectory
            states = generate_trajectory(shape.value, seed=42)
            window = compute_trajectory_window(states)

            # Classify (should match the intended shape)
            classified = classify_trajectory(window)
            assert classified == shape, (
                f"Generated trajectory for {shape.value} classified as {classified.value}"
            )

            # Build training example
            example = build_training_example(shape.value, window, seed=42)
            assert example.shape == shape.value
            assert len(example.eisv_tokens) > 0

            # Format as chat messages
            messages = format_chat_messages(example)
            assert len(messages) == 3, (
                f"Expected 3 messages for {shape.value}, got {len(messages)}"
            )

            # Verify shape appears in user message content
            user_content = messages[1]["content"]
            assert shape.value in user_content, (
                f"Shape {shape.value} not found in user message"
            )

    def test_prepare_and_evaluate(self):
        """Prepare synthetic data, parse ground truth, evaluate.

        Rule-based labels (without feedback) produce moderate coherence
        (~0.48), which should be above 0.4 and have near-perfect valid rate.
        """
        train, val, test = prepare_training_data(
            real_records=[], min_per_shape=10, seed=42,
        )

        # Build predictions from training data
        predictions = []
        for item in train:
            messages = item["messages"]
            assistant_text = messages[2]["content"]
            parsed = parse_model_output(assistant_text)
            predictions.append({
                "shape": item["shape"],
                "parsed": parsed,
                "expected_pattern": item["pattern"],
            })

        results = evaluate_predictions(predictions)

        # Rule-based labels produce moderate coherence (above random)
        assert results.mean_coherence > 0.4, (
            f"Coherence {results.mean_coherence:.4f} should be > 0.4"
        )
        # All outputs should be valid since they come from our own formatter
        assert results.valid_rate > 0.9, (
            f"Valid rate {results.valid_rate:.4f} should be > 0.9"
        )

    def test_gate1_on_rule_labels(self):
        """Rule-based labels should FAIL Gate 1 (coherence ~0.48 < 0.933).

        This is expected: the gate is designed for a fine-tuned model,
        not raw rule-based generation. We verify that:
        1. Gate 1 correctly identifies this as failing
        2. Coherence is above random baseline (> 0.3)
        """
        train, val, test = prepare_training_data(
            real_records=[], min_per_shape=10, seed=42,
        )

        predictions = []
        for item in train:
            messages = item["messages"]
            assistant_text = messages[2]["content"]
            parsed = parse_model_output(assistant_text)
            predictions.append({
                "shape": item["shape"],
                "parsed": parsed,
            })

        results = evaluate_predictions(predictions)

        # Gate 1 should FAIL for rule-based labels
        passed, reasons = check_gate1(results)
        assert passed is False, (
            "Gate 1 should fail for rule-based labels without feedback tuning"
        )

        # But coherence should be above random baseline
        assert results.mean_coherence > 0.3, (
            f"Coherence {results.mean_coherence:.4f} should be above "
            f"random baseline (> 0.3)"
        )
