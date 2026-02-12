"""Tests for dynamics-emergent expression generator."""

import pytest
from eisv_lumen.shapes.expression_generator import (
    ExpressionPattern,
    ExpressionGenerator,
    SHAPE_PATTERN_WEIGHTS,
    INQUIRY_TOKENS,
)
from eisv_lumen.eval.baseline import ALL_TOKENS
from eisv_lumen.eval.metrics import SHAPE_TOKEN_AFFINITY
from eisv_lumen.shapes.shape_classes import TrajectoryShape


class TestExpressionPatternEnum:
    def test_has_five_patterns(self):
        assert len(ExpressionPattern) == 5

    def test_all_string_values(self):
        for p in ExpressionPattern:
            assert isinstance(p.value, str)


class TestShapePatternWeights:
    def test_all_nine_shapes_present(self):
        for shape in TrajectoryShape:
            assert shape.value in SHAPE_PATTERN_WEIGHTS

    def test_weights_sum_to_one(self):
        for shape, weights in SHAPE_PATTERN_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.01, f"{shape} weights sum to {total}"

    def test_all_patterns_in_each_shape(self):
        for shape, weights in SHAPE_PATTERN_WEIGHTS.items():
            for pattern in ExpressionPattern:
                assert pattern.value in weights, f"{shape} missing {pattern.value}"


class TestExpressionGenerator:
    def test_generate_returns_list_of_tokens(self):
        gen = ExpressionGenerator(seed=42)
        result = gen.generate("settled_presence")
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)
        assert all(t in ALL_TOKENS for t in result)

    def test_generate_length_varies(self):
        gen = ExpressionGenerator(seed=42)
        lengths = set()
        for _ in range(100):
            result = gen.generate("rising_entropy")
            lengths.add(len(result))
        # Should produce at least 2 different lengths
        assert len(lengths) >= 2

    def test_deterministic_with_seed(self):
        g1 = ExpressionGenerator(seed=99)
        g2 = ExpressionGenerator(seed=99)
        r1 = [g1.generate("settled_presence") for _ in range(10)]
        r2 = [g2.generate("settled_presence") for _ in range(10)]
        assert r1 == r2

    def test_affine_tokens_favored(self):
        gen = ExpressionGenerator(seed=42)
        affine = set(SHAPE_TOKEN_AFFINITY["settled_presence"])
        tokens = []
        for _ in range(200):
            tokens.extend(gen.generate("settled_presence"))
        affine_count = sum(1 for t in tokens if t in affine)
        # With weight 3.0 vs 1.0, affine should be >40% of total
        assert affine_count / len(tokens) > 0.4

    def test_question_pattern_ends_with_inquiry(self):
        gen = ExpressionGenerator(seed=42)
        questions_found = 0
        for _ in range(500):
            result = gen.generate("drift_dissonance")  # 40% question weight
            if len(result) == 2 and result[-1] in INQUIRY_TOKENS:
                questions_found += 1
        assert questions_found > 0

    def test_repetition_pattern_has_repeated_token(self):
        gen = ExpressionGenerator(seed=42)
        reps_found = 0
        for _ in range(500):
            result = gen.generate("settled_presence")  # 15% repetition weight
            if len(result) == 2 and result[0] == result[1]:
                reps_found += 1
        assert reps_found > 0


class TestFeedbackWeightUpdate:
    def test_positive_feedback_increases_weight(self):
        gen = ExpressionGenerator(seed=42)
        w_before = gen.get_weights("settled_presence")["~stillness~"]
        gen.update_weights("settled_presence", ["~stillness~"], score=0.9)
        w_after = gen.get_weights("settled_presence")["~stillness~"]
        assert w_after > w_before

    def test_negative_feedback_decreases_weight(self):
        gen = ExpressionGenerator(seed=42)
        w_before = gen.get_weights("settled_presence")["~stillness~"]
        gen.update_weights("settled_presence", ["~stillness~"], score=0.1)
        w_after = gen.get_weights("settled_presence")["~stillness~"]
        assert w_after < w_before

    def test_weight_clamping(self):
        gen = ExpressionGenerator(seed=42)
        # Push weight down many times
        for _ in range(200):
            gen.update_weights("settled_presence", ["~stillness~"], score=0.0)
        w = gen.get_weights("settled_presence")["~stillness~"]
        assert w >= 0.1

        # Push weight up many times
        for _ in range(200):
            gen.update_weights("settled_presence", ["~warmth~"], score=1.0)
        w = gen.get_weights("settled_presence")["~warmth~"]
        assert w <= 10.0

    def test_neutral_feedback_no_change(self):
        gen = ExpressionGenerator(seed=42)
        w_before = gen.get_weights("settled_presence")["~stillness~"]
        gen.update_weights("settled_presence", ["~stillness~"], score=0.5)
        w_after = gen.get_weights("settled_presence")["~stillness~"]
        assert abs(w_after - w_before) < 1e-9

    def test_unknown_shape_no_crash(self):
        gen = ExpressionGenerator(seed=42)
        gen.update_weights("nonexistent_shape", ["~warmth~"], score=0.8)  # should not crash
