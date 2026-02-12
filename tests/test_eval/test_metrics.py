"""Tests for evaluation metrics."""

import pytest
from eisv_lumen.eval.metrics import (
    SHAPE_TOKEN_AFFINITY,
    shape_classification_accuracy,
    expression_trajectory_coherence,
    vocabulary_diversity_per_shape,
)


class TestShapeTokenAffinity:
    def test_has_all_nine_shapes(self):
        from eisv_lumen.shapes.shape_classes import TrajectoryShape
        for shape in TrajectoryShape:
            assert shape.value in SHAPE_TOKEN_AFFINITY

    def test_all_values_are_lists(self):
        for shape, tokens in SHAPE_TOKEN_AFFINITY.items():
            assert isinstance(tokens, list)
            assert all(isinstance(t, str) for t in tokens)


class TestShapeClassificationAccuracy:
    def test_perfect_accuracy(self):
        pred = ["settled_presence", "rising_entropy", "convergence"]
        actual = ["settled_presence", "rising_entropy", "convergence"]
        assert shape_classification_accuracy(pred, actual) == 1.0

    def test_zero_accuracy(self):
        pred = ["settled_presence", "rising_entropy"]
        actual = ["convergence", "void_rising"]
        assert shape_classification_accuracy(pred, actual) == 0.0

    def test_partial_accuracy(self):
        pred = ["settled_presence", "rising_entropy", "convergence", "void_rising"]
        actual = ["settled_presence", "void_rising", "convergence", "void_rising"]
        # Matches: index 0, 2, 3 = 3/4 = 0.75
        assert shape_classification_accuracy(pred, actual) == 0.75

    def test_empty_lists(self):
        assert shape_classification_accuracy([], []) == 0.0


class TestExpressionTrajectoryCoherence:
    def test_all_affine_tokens(self):
        # settled_presence affinity: ~stillness~, ~holding~, ~resonance~, ~deep_listening~
        score = expression_trajectory_coherence(
            "settled_presence",
            ["~stillness~", "~holding~", "~resonance~"],
        )
        assert score == 1.0

    def test_no_affine_tokens(self):
        score = expression_trajectory_coherence(
            "settled_presence",
            ["~reaching~", "~emergence~"],
        )
        assert score == 0.0

    def test_mixed_tokens(self):
        score = expression_trajectory_coherence(
            "settled_presence",
            ["~stillness~", "~reaching~"],  # 1 affine out of 2
        )
        assert score == pytest.approx(0.5)

    def test_empty_tokens(self):
        assert expression_trajectory_coherence("settled_presence", []) == 0.0

    def test_unknown_shape(self):
        assert expression_trajectory_coherence("nonexistent", ["~warmth~"]) == 0.0


class TestVocabularyDiversityPerShape:
    def test_single_shape_all_unique(self):
        records = [
            {"shape": "settled_presence", "tokens": ["~stillness~", "~holding~", "~resonance~"]},
        ]
        result = vocabulary_diversity_per_shape(records)
        assert result["settled_presence"] == pytest.approx(1.0)

    def test_single_shape_repeated(self):
        records = [
            {"shape": "settled_presence", "tokens": ["~stillness~", "~stillness~", "~holding~"]},
        ]
        result = vocabulary_diversity_per_shape(records)
        # 2 unique / 3 total
        assert result["settled_presence"] == pytest.approx(2.0 / 3.0)

    def test_multiple_shapes(self):
        records = [
            {"shape": "settled_presence", "tokens": ["~stillness~"]},
            {"shape": "rising_entropy", "tokens": ["~ripple~", "~ripple~"]},
        ]
        result = vocabulary_diversity_per_shape(records)
        assert result["settled_presence"] == pytest.approx(1.0)
        assert result["rising_entropy"] == pytest.approx(0.5)

    def test_empty_records(self):
        result = vocabulary_diversity_per_shape([])
        assert result == {}

    def test_shape_with_no_tokens(self):
        records = [{"shape": "convergence", "tokens": []}]
        result = vocabulary_diversity_per_shape(records)
        assert result["convergence"] == 0.0
