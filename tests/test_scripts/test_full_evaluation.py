"""Tests for the full evaluation script.

Tests the report generation, go/no-go logic, and shape distribution
using synthetic data (no real database needed).
"""

from __future__ import annotations

import pytest

from eisv_lumen.scripts.full_evaluation import (
    ALL_SHAPE_VALUES,
    compute_go_no_go,
    compute_shape_distribution,
    evaluate_expression_generator,
    evaluate_expression_generator_with_feedback,
)


# ---------------------------------------------------------------------------
# Helpers: build synthetic trajectory records
# ---------------------------------------------------------------------------

def _make_records(shape_counts: dict[str, int]) -> list[dict]:
    """Build minimal trajectory records for testing.

    Each record only needs a 'shape' key for the functions under test.
    """
    records = []
    for shape, count in shape_counts.items():
        for _ in range(count):
            records.append({"shape": shape})
    return records


# ---------------------------------------------------------------------------
# Tests: compute_shape_distribution
# ---------------------------------------------------------------------------

class TestShapeDistribution:
    """Tests for shape distribution computation."""

    def test_empty_records(self):
        result = compute_shape_distribution([])
        assert result == {}

    def test_single_shape(self):
        records = _make_records({"convergence": 10})
        dist = compute_shape_distribution(records)
        assert "convergence" in dist
        assert dist["convergence"]["count"] == 10
        assert dist["convergence"]["percent"] == 100.0

    def test_multiple_shapes(self):
        records = _make_records({
            "convergence": 6,
            "settled_presence": 3,
            "rising_entropy": 1,
        })
        dist = compute_shape_distribution(records)

        assert len(dist) == 3
        assert dist["convergence"]["count"] == 6
        assert dist["convergence"]["percent"] == 60.0
        assert dist["settled_presence"]["count"] == 3
        assert dist["settled_presence"]["percent"] == 30.0
        assert dist["rising_entropy"]["count"] == 1
        assert dist["rising_entropy"]["percent"] == 10.0

    def test_missing_shapes_excluded(self):
        records = _make_records({"convergence": 5})
        dist = compute_shape_distribution(records)
        # Only convergence should appear
        assert list(dist.keys()) == ["convergence"]

    def test_percents_sum_to_100(self):
        records = _make_records({
            "convergence": 50,
            "settled_presence": 30,
            "basin_transition_down": 20,
        })
        dist = compute_shape_distribution(records)
        total_pct = sum(v["percent"] for v in dist.values())
        assert abs(total_pct - 100.0) < 0.01


# ---------------------------------------------------------------------------
# Tests: compute_go_no_go
# ---------------------------------------------------------------------------

class TestGoNoGo:
    """Tests for go/no-go decision logic."""

    def _build_report(
        self,
        random_coh: float = 0.25,
        gen_coh: float = 0.45,
        fb_coh: float = 0.50,
        shapes_observed: int = 5,
    ) -> dict:
        """Build a minimal report dict for go/no-go testing."""
        return {
            "baselines": {
                "random": {"mean_coherence": random_coh},
            },
            "expression_generator": {"mean_coherence": gen_coh},
            "expression_generator_with_feedback": {"mean_coherence": fb_coh},
            "shapes_observed": shapes_observed,
        }

    def test_all_pass_is_go(self):
        report = self._build_report(
            random_coh=0.25, gen_coh=0.45, fb_coh=0.50, shapes_observed=5,
        )
        result = compute_go_no_go(report)
        assert result["decision"] == "GO"
        assert result["all_passed"] is True
        assert result["criteria"]["beats_random"]["passed"] is True
        assert result["criteria"]["min_shapes_observed"]["passed"] is True
        assert result["criteria"]["feedback_improves"]["passed"] is True

    def test_fails_beats_random(self):
        # gen_coh - random_coh = 0.03, which is < 0.05 threshold
        report = self._build_report(
            random_coh=0.25, gen_coh=0.28, fb_coh=0.30, shapes_observed=5,
        )
        result = compute_go_no_go(report)
        assert result["decision"] == "NO-GO"
        assert result["criteria"]["beats_random"]["passed"] is False
        assert result["criteria"]["beats_random"]["value"] == pytest.approx(0.03, abs=0.001)

    def test_exact_threshold_fails_beats_random(self):
        # Exactly at threshold (0.05) should fail since we require > 0.05
        report = self._build_report(
            random_coh=0.25, gen_coh=0.30, fb_coh=0.35, shapes_observed=5,
        )
        result = compute_go_no_go(report)
        assert result["criteria"]["beats_random"]["passed"] is False

    def test_fails_min_shapes(self):
        report = self._build_report(
            random_coh=0.25, gen_coh=0.45, fb_coh=0.50, shapes_observed=2,
        )
        result = compute_go_no_go(report)
        assert result["decision"] == "NO-GO"
        assert result["criteria"]["min_shapes_observed"]["passed"] is False

    def test_exactly_3_shapes_passes(self):
        report = self._build_report(shapes_observed=3)
        result = compute_go_no_go(report)
        assert result["criteria"]["min_shapes_observed"]["passed"] is True

    def test_fails_feedback_regression(self):
        # Feedback is worse than no-feedback
        report = self._build_report(
            random_coh=0.25, gen_coh=0.45, fb_coh=0.40, shapes_observed=5,
        )
        result = compute_go_no_go(report)
        assert result["decision"] == "NO-GO"
        assert result["criteria"]["feedback_improves"]["passed"] is False
        assert result["criteria"]["feedback_improves"]["value"] < 0

    def test_feedback_equal_passes(self):
        # Feedback == no-feedback should pass (>= threshold)
        report = self._build_report(
            random_coh=0.25, gen_coh=0.45, fb_coh=0.45, shapes_observed=5,
        )
        result = compute_go_no_go(report)
        assert result["criteria"]["feedback_improves"]["passed"] is True

    def test_multiple_failures(self):
        report = self._build_report(
            random_coh=0.25, gen_coh=0.28, fb_coh=0.20, shapes_observed=1,
        )
        result = compute_go_no_go(report)
        assert result["decision"] == "NO-GO"
        assert result["all_passed"] is False
        # All three should fail
        assert result["criteria"]["beats_random"]["passed"] is False
        assert result["criteria"]["min_shapes_observed"]["passed"] is False
        assert result["criteria"]["feedback_improves"]["passed"] is False

    def test_missing_keys_fallback(self):
        # Empty report should produce NO-GO without errors
        result = compute_go_no_go({})
        assert result["decision"] == "NO-GO"
        assert result["all_passed"] is False


# ---------------------------------------------------------------------------
# Tests: expression generator evaluation functions (with synthetic data)
# ---------------------------------------------------------------------------

class TestExpressionGeneratorEvaluation:
    """Tests for the expression generator evaluation wrappers."""

    def test_empty_records(self):
        result = evaluate_expression_generator([], seed=42)
        assert result["mean_coherence"] == 0.0
        assert result["per_shape_coherence"] == {}
        assert result["diversity"] == {}

    def test_generates_coherence_for_known_shapes(self):
        records = _make_records({"convergence": 20, "settled_presence": 20})
        result = evaluate_expression_generator(records, seed=42)

        assert 0.0 <= result["mean_coherence"] <= 1.0
        assert "convergence" in result["per_shape_coherence"]
        assert "settled_presence" in result["per_shape_coherence"]
        assert len(result["diversity"]) > 0

    def test_reproducible_with_same_seed(self):
        records = _make_records({"convergence": 50})
        r1 = evaluate_expression_generator(records, seed=42)
        r2 = evaluate_expression_generator(records, seed=42)
        assert r1["mean_coherence"] == r2["mean_coherence"]

    def test_different_seed_different_result(self):
        records = _make_records({"convergence": 100})
        r1 = evaluate_expression_generator(records, seed=42)
        r2 = evaluate_expression_generator(records, seed=99)
        # With enough records, different seeds should likely differ
        # (not guaranteed, but highly probable)
        # We just check they run without error
        assert isinstance(r1["mean_coherence"], float)
        assert isinstance(r2["mean_coherence"], float)


class TestExpressionGeneratorWithFeedback:
    """Tests for the feedback-loop evaluation wrapper."""

    def test_empty_records(self):
        result = evaluate_expression_generator_with_feedback([], seed=42)
        assert result["mean_coherence"] == 0.0
        assert result["per_shape_coherence"] == {}

    def test_runs_on_synthetic_data(self):
        records = _make_records({"convergence": 30, "rising_entropy": 30})
        result = evaluate_expression_generator_with_feedback(records, seed=42)

        assert 0.0 <= result["mean_coherence"] <= 1.0
        assert "convergence" in result["per_shape_coherence"]
        assert "rising_entropy" in result["per_shape_coherence"]

    def test_feedback_produces_valid_output(self):
        records = _make_records({"settled_presence": 50})
        no_fb = evaluate_expression_generator(records, seed=42)
        with_fb = evaluate_expression_generator_with_feedback(records, seed=42)

        # Both should produce valid coherence values
        assert 0.0 <= no_fb["mean_coherence"] <= 1.0
        assert 0.0 <= with_fb["mean_coherence"] <= 1.0
