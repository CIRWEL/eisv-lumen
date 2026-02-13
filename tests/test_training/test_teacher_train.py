"""Tests for eisv_lumen.training.teacher_train — teacher training pipeline."""

from __future__ import annotations

import pytest

from eisv_lumen.shapes.shape_classes import TrajectoryShape
from eisv_lumen.training.teacher_train import (
    VALID_PATTERNS,
    EISV_TOKEN_SET,
    LUMEN_TOKEN_SET,
    OutputParseResult,
    prepare_training_data,
    parse_model_output,
    validate_output,
)


# ---------------------------------------------------------------------------
# TestPrepareTrainingData
# ---------------------------------------------------------------------------


class TestPrepareTrainingData:
    def test_returns_hf_dataset_dicts(self):
        """Each returned dict should have 'text', 'shape', and 'messages' keys."""
        train, val, test = prepare_training_data([], min_per_shape=10, seed=42)
        # train must be non-empty; val/test may vary but combined should be > 0
        assert len(train) > 0
        assert len(train) + len(val) + len(test) == 9 * 10  # 9 shapes * 10 each
        for split in (train, val, test):
            for item in split:
                assert isinstance(item, dict)
                assert "text" in item, f"Missing 'text' key in {item.keys()}"
                assert "shape" in item, f"Missing 'shape' key in {item.keys()}"
                assert "messages" in item, f"Missing 'messages' key in {item.keys()}"

    def test_all_shapes_in_train(self):
        """Training set should contain all 9 trajectory shapes."""
        train, _, _ = prepare_training_data([], min_per_shape=5, seed=42)
        shapes_present = {item["shape"] for item in train}
        expected = {s.value for s in TrajectoryShape}
        assert shapes_present == expected, (
            f"Missing shapes: {expected - shapes_present}"
        )


# ---------------------------------------------------------------------------
# TestParseModelOutput
# ---------------------------------------------------------------------------


class TestParseModelOutput:
    def test_valid_output(self):
        """A well-formed output should parse correctly with valid=True."""
        text = (
            "EISV_TOKENS: ~warmth~ ~stillness~\n"
            "LUMEN_TOKENS: warm quiet\n"
            "PATTERN: PAIR"
        )
        result = parse_model_output(text)
        assert result.valid is True
        assert result.eisv_tokens == ["~warmth~", "~stillness~"]
        assert result.lumen_tokens == ["warm", "quiet"]
        assert result.pattern == "PAIR"

    def test_single_token(self):
        """A single-token output should parse correctly."""
        text = (
            "EISV_TOKENS: ~resonance~\n"
            "LUMEN_TOKENS: with\n"
            "PATTERN: SINGLE"
        )
        result = parse_model_output(text)
        assert result.valid is True
        assert result.eisv_tokens == ["~resonance~"]
        assert result.lumen_tokens == ["with"]
        assert result.pattern == "SINGLE"

    def test_malformed_output(self):
        """Garbage text should return valid=False and empty lists."""
        result = parse_model_output("this is garbage text with no structure")
        assert result.valid is False
        assert result.eisv_tokens == []
        assert result.lumen_tokens == []
        assert result.pattern == ""

    def test_partial_output(self):
        """Missing LUMEN_TOKENS should result in valid=False."""
        text = (
            "EISV_TOKENS: ~warmth~\n"
            "PATTERN: SINGLE"
        )
        result = parse_model_output(text)
        assert result.valid is False


# ---------------------------------------------------------------------------
# TestValidateOutput
# ---------------------------------------------------------------------------


class TestValidateOutput:
    def test_valid_tokens(self):
        """All tokens in vocabulary and pattern valid should return True."""
        result = OutputParseResult(
            eisv_tokens=["~warmth~", "~stillness~"],
            lumen_tokens=["warm", "quiet"],
            pattern="PAIR",
            valid=True,
        )
        assert validate_output(result) is True

    def test_invalid_eisv_token(self):
        """An unrecognized EISV token should cause validation failure."""
        result = OutputParseResult(
            eisv_tokens=["~warmth~", "~nonexistent_token~"],
            lumen_tokens=["warm", "quiet"],
            pattern="PAIR",
            valid=True,
        )
        assert validate_output(result) is False

    def test_invalid_pattern(self):
        """An unrecognized pattern should cause validation failure."""
        result = OutputParseResult(
            eisv_tokens=["~warmth~"],
            lumen_tokens=["warm"],
            pattern="INVALID_PATTERN",
            valid=True,
        )
        assert validate_output(result) is False

    def test_invalid_lumen_token(self):
        """An unrecognized Lumen token should cause validation failure."""
        result = OutputParseResult(
            eisv_tokens=["~warmth~"],
            lumen_tokens=["nonexistent_lumen_token"],
            pattern="SINGLE",
            valid=True,
        )
        assert validate_output(result) is False
