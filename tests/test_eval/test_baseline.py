"""Tests for baseline evaluation."""

import pytest
from eisv_lumen.eval.baseline import (
    ALL_TOKENS,
    BaselineCondition,
    generate_baseline_tokens,
    evaluate_baseline,
    format_eisv_prompt,
)
from eisv_lumen.eval.metrics import SHAPE_TOKEN_AFFINITY


class TestAllTokens:
    def test_has_fifteen_tokens(self):
        assert len(ALL_TOKENS) == 15

    def test_all_wrapped_in_tildes(self):
        for token in ALL_TOKENS:
            assert token.startswith("~") and token.endswith("~")


class TestGenerateBaselineTokens:
    def test_random_returns_n_tokens(self):
        import random
        rng = random.Random(42)
        tokens = generate_baseline_tokens("settled_presence", BaselineCondition.RANDOM, n_tokens=5, rng=rng)
        assert len(tokens) == 5
        assert all(t in ALL_TOKENS for t in tokens)

    def test_shape_matched_all_affine(self):
        import random
        rng = random.Random(42)
        tokens = generate_baseline_tokens("settled_presence", BaselineCondition.SHAPE_MATCHED, n_tokens=10, rng=rng)
        affinity = SHAPE_TOKEN_AFFINITY["settled_presence"]
        assert all(t in affinity for t in tokens)

    def test_prompt_conditioned_mixed(self):
        import random
        rng = random.Random(42)
        # With enough tokens, should have mix of affine and random
        tokens = generate_baseline_tokens("settled_presence", BaselineCondition.PROMPT_CONDITIONED, n_tokens=100, rng=rng)
        affinity = set(SHAPE_TOKEN_AFFINITY["settled_presence"])
        affine_count = sum(1 for t in tokens if t in affinity)
        # Should be roughly 70% +/- margin
        assert 50 < affine_count < 90  # generous bounds for stochastic

    def test_unknown_shape_falls_back_to_random(self):
        import random
        rng = random.Random(42)
        tokens = generate_baseline_tokens("nonexistent", BaselineCondition.SHAPE_MATCHED, n_tokens=3, rng=rng)
        assert len(tokens) == 3
        assert all(t in ALL_TOKENS for t in tokens)

    def test_deterministic_with_same_seed(self):
        import random
        t1 = generate_baseline_tokens("rising_entropy", BaselineCondition.RANDOM, n_tokens=5, rng=random.Random(99))
        t2 = generate_baseline_tokens("rising_entropy", BaselineCondition.RANDOM, n_tokens=5, rng=random.Random(99))
        assert t1 == t2


class TestEvaluateBaseline:
    def test_returns_expected_keys(self):
        records = [{"shape": "settled_presence"}, {"shape": "rising_entropy"}, {"shape": "convergence"}]
        result = evaluate_baseline(records, BaselineCondition.RANDOM)
        assert "condition" in result
        assert "mean_coherence" in result
        assert "per_shape_coherence" in result
        assert "diversity" in result
        assert "n_records" in result
        assert result["n_records"] == 3
        assert result["condition"] == "random"

    def test_shape_matched_high_coherence(self):
        records = [{"shape": "settled_presence"}] * 10
        result = evaluate_baseline(records, BaselineCondition.SHAPE_MATCHED, seed=42)
        assert result["mean_coherence"] == 1.0

    def test_random_lower_coherence(self):
        records = [{"shape": "settled_presence"}] * 50
        result_random = evaluate_baseline(records, BaselineCondition.RANDOM, seed=42)
        result_matched = evaluate_baseline(records, BaselineCondition.SHAPE_MATCHED, seed=42)
        assert result_random["mean_coherence"] < result_matched["mean_coherence"]

    def test_empty_records(self):
        result = evaluate_baseline([], BaselineCondition.RANDOM)
        assert result["mean_coherence"] == 0.0
        assert result["n_records"] == 0


class TestFormatEisvPrompt:
    def test_contains_values(self):
        prompt = format_eisv_prompt(
            {"E": 0.72, "I": 0.65, "S": 0.31, "V": 0.08},
            "settled_presence",
        )
        assert "0.72" in prompt
        assert "settled_presence" in prompt

    def test_is_string(self):
        prompt = format_eisv_prompt({"E": 0.5, "I": 0.5, "S": 0.5, "V": 0.5}, "convergence")
        assert isinstance(prompt, str)
        assert len(prompt) > 0
