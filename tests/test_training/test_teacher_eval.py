"""Tests for eisv_lumen.training.teacher_eval — teacher evaluation and Gate 1."""

from __future__ import annotations

import pytest

from eisv_lumen.training.teacher_train import OutputParseResult
from eisv_lumen.training.teacher_eval import (
    GATE1_COHERENCE_THRESHOLD,
    GATE1_VALID_RATE_THRESHOLD,
    EvalResults,
    evaluate_predictions,
    check_gate1,
)


def _make_prediction(
    shape: str,
    eisv_tokens: list[str],
    lumen_tokens: list[str],
    pattern: str,
    valid: bool = True,
    expected_pattern: str | None = None,
) -> dict:
    """Helper to build a prediction dict."""
    parsed = OutputParseResult(
        eisv_tokens=eisv_tokens,
        lumen_tokens=lumen_tokens,
        pattern=pattern,
        valid=valid,
    )
    pred = {"shape": shape, "parsed": parsed}
    if expected_pattern is not None:
        pred["expected_pattern"] = expected_pattern
    return pred


# ---------------------------------------------------------------------------
# TestEvaluatePredictions
# ---------------------------------------------------------------------------


class TestEvaluatePredictions:
    def test_perfect_predictions(self):
        """All tokens affine to their shape should yield coherence=1.0."""
        # Use settled_presence with its affine tokens
        predictions = [
            _make_prediction(
                shape="settled_presence",
                eisv_tokens=["~stillness~", "~holding~"],
                lumen_tokens=["quiet", "here"],
                pattern="PAIR",
                expected_pattern="PAIR",
            )
            for _ in range(10)
        ]
        results = evaluate_predictions(predictions)
        assert results.mean_coherence == 1.0
        assert results.valid_rate == 1.0
        assert results.n_total == 10
        assert results.n_valid == 10

    def test_mixed_predictions(self):
        """Half valid, half invalid should yield valid_rate=0.5."""
        valid_preds = [
            _make_prediction(
                shape="settled_presence",
                eisv_tokens=["~stillness~"],
                lumen_tokens=["quiet"],
                pattern="SINGLE",
            )
            for _ in range(5)
        ]
        invalid_preds = [
            _make_prediction(
                shape="settled_presence",
                eisv_tokens=[],
                lumen_tokens=[],
                pattern="",
                valid=False,
            )
            for _ in range(5)
        ]
        results = evaluate_predictions(valid_preds + invalid_preds)
        assert results.valid_rate == 0.5
        assert results.n_total == 10
        assert results.n_valid == 5

    def test_pattern_accuracy(self):
        """Pattern accuracy should reflect matches with expected_pattern."""
        predictions = [
            _make_prediction(
                shape="settled_presence",
                eisv_tokens=["~stillness~"],
                lumen_tokens=["quiet"],
                pattern="SINGLE",
                expected_pattern="SINGLE",
            ),
            _make_prediction(
                shape="rising_entropy",
                eisv_tokens=["~ripple~", "~emergence~"],
                lumen_tokens=["busy", "more"],
                pattern="PAIR",
                expected_pattern="TRIPLE",  # mismatch
            ),
        ]
        results = evaluate_predictions(predictions)
        assert results.pattern_accuracy == 0.5

    def test_empty_predictions(self):
        """Empty predictions should return zero metrics."""
        results = evaluate_predictions([])
        assert results.mean_coherence == 0.0
        assert results.valid_rate == 0.0
        assert results.n_total == 0
        assert results.n_valid == 0


# ---------------------------------------------------------------------------
# TestCheckGate1
# ---------------------------------------------------------------------------


class TestCheckGate1:
    def test_passes(self):
        """Results above both thresholds should pass Gate 1."""
        results = EvalResults(
            mean_coherence=0.95,
            valid_rate=0.95,
            pattern_accuracy=0.8,
            n_total=100,
            n_valid=95,
            per_shape_coherence={"settled_presence": 0.95},
            diversity=0.7,
        )
        passed, reasons = check_gate1(results)
        assert passed is True
        assert len(reasons) == 0

    def test_fails_low_coherence(self):
        """Coherence below threshold should fail Gate 1."""
        results = EvalResults(
            mean_coherence=0.80,  # below 0.933
            valid_rate=0.95,
            pattern_accuracy=0.8,
            n_total=100,
            n_valid=95,
            per_shape_coherence={},
            diversity=0.7,
        )
        passed, reasons = check_gate1(results)
        assert passed is False
        assert any("coherence" in r.lower() for r in reasons)

    def test_fails_low_valid_rate(self):
        """Valid rate below threshold should fail Gate 1."""
        results = EvalResults(
            mean_coherence=0.95,
            valid_rate=0.85,  # below 0.90
            pattern_accuracy=0.8,
            n_total=100,
            n_valid=85,
            per_shape_coherence={},
            diversity=0.7,
        )
        passed, reasons = check_gate1(results)
        assert passed is False
        assert any("valid" in r.lower() for r in reasons)
