"""Tests for eisv_lumen.training.data_prep — training data formatter."""

from __future__ import annotations

import pytest

from eisv_lumen.shapes.shape_classes import TrajectoryShape
from eisv_lumen.bridge.lumen_bridge import translate_expression
from eisv_lumen.extract.derivatives import compute_trajectory_window
from eisv_lumen.synthetic.trajectory_generator import generate_trajectory
from eisv_lumen.training.data_prep import (
    TrainingExample,
    format_trajectory_input,
    format_expression_output,
    build_training_example,
    _infer_pattern,
)


def _make_window(shape: str = "settled_presence", seed: int = 42) -> dict:
    """Helper: generate a trajectory window for a given shape."""
    states = generate_trajectory(shape, seed=seed)
    return compute_trajectory_window(states)


# ---------------------------------------------------------------------------
# TestFormatTrajectoryInput
# ---------------------------------------------------------------------------


class TestFormatTrajectoryInput:
    def test_basic_format(self):
        """Output should contain SHAPE, WINDOW, MEAN_EISV, DERIVATIVES, SECOND_DERIVATIVES sections."""
        window = _make_window("settled_presence")
        text = format_trajectory_input("settled_presence", window)
        assert "SHAPE: settled_presence" in text
        assert "E=" in text
        assert "dE=" in text
        assert "d2E=" in text

    def test_contains_numeric_values(self):
        """Mean values should be numeric floats."""
        window = _make_window("rising_entropy")
        text = format_trajectory_input("rising_entropy", window)
        # Extract E= value and check it parses as float
        for line in text.split("\n"):
            if "E=" in line and "dE=" not in line and "d2E=" not in line:
                # Find the E= portion
                parts = line.split()
                for p in parts:
                    if p.startswith("E="):
                        val = float(p.split("=")[1])
                        assert 0.0 <= val <= 1.0
                        break

    def test_window_metadata(self):
        """Output should include n_states and duration."""
        window = _make_window("settled_presence")
        text = format_trajectory_input("settled_presence", window)
        assert "n_states=" in text
        assert "duration=" in text


# ---------------------------------------------------------------------------
# TestFormatExpressionOutput
# ---------------------------------------------------------------------------


class TestFormatExpressionOutput:
    def test_basic_format(self):
        """Output should contain EISV_TOKENS, LUMEN_TOKENS, PATTERN lines."""
        text = format_expression_output(
            ["~warmth~", "~stillness~"],
            ["warm", "quiet"],
            "PAIR",
        )
        assert "EISV_TOKENS:" in text
        assert "LUMEN_TOKENS:" in text
        assert "PATTERN:" in text
        assert "~warmth~ ~stillness~" in text

    def test_single_token(self):
        """Single token expressions should format correctly."""
        text = format_expression_output(
            ["~warmth~"],
            ["warm"],
            "SINGLE",
        )
        assert "~warmth~" in text
        assert "warm" in text
        assert "SINGLE" in text

    def test_triple_token(self):
        """Triple token expressions should format correctly."""
        text = format_expression_output(
            ["~warmth~", "~stillness~", "~resonance~"],
            ["warm", "quiet", "with"],
            "TRIPLE",
        )
        assert "~warmth~ ~stillness~ ~resonance~" in text
        assert "TRIPLE" in text


# ---------------------------------------------------------------------------
# TestBuildTrainingExample
# ---------------------------------------------------------------------------


class TestBuildTrainingExample:
    def test_returns_training_example(self):
        """build_training_example should return a TrainingExample dataclass."""
        window = _make_window("settled_presence")
        example = build_training_example("settled_presence", window, seed=123)
        assert isinstance(example, TrainingExample)
        assert example.shape == "settled_presence"
        assert len(example.eisv_tokens) >= 1
        assert len(example.input_text) > 0
        assert len(example.output_text) > 0

    def test_deterministic_with_seed(self):
        """Same seed should produce same output."""
        window = _make_window("rising_entropy", seed=99)
        ex1 = build_training_example("rising_entropy", window, seed=42)
        ex2 = build_training_example("rising_entropy", window, seed=42)
        assert ex1.eisv_tokens == ex2.eisv_tokens
        assert ex1.lumen_tokens == ex2.lumen_tokens
        assert ex1.pattern == ex2.pattern

    def test_lumen_tokens_match_bridge(self):
        """Lumen tokens should match what translate_expression produces."""
        window = _make_window("basin_transition_up")
        example = build_training_example("basin_transition_up", window, seed=77)
        expected_lumen = translate_expression(example.eisv_tokens)
        assert example.lumen_tokens == expected_lumen
