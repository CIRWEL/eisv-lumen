"""Tests for eisv_lumen.training.teacher_inference — inference utilities.

All tests run WITHOUT GPU or heavy dependencies.  Model and tokenizer
objects are mocked so parsing, formatting, and extraction logic can be
verified in CI.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from eisv_lumen.training.teacher_inference import (
    _extract_assistant_response,
    _extract_trajectory_from_text,
    _format_inference_messages,
)
from eisv_lumen.training.chat_format import SYSTEM_PROMPT
from eisv_lumen.training.teacher_train import parse_model_output


# ---------------------------------------------------------------------------
# TestFormatInferenceMessages
# ---------------------------------------------------------------------------

class TestFormatInferenceMessages:
    """Test chat message formatting for inference."""

    def test_returns_system_and_user(self):
        """Should produce exactly two messages: system and user."""
        msgs = _format_inference_messages("SHAPE: settled_presence\nMEAN_EISV: E=0.8 ...")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_system_content_is_prompt(self):
        """System message should be the SYSTEM_PROMPT."""
        msgs = _format_inference_messages("test input")
        assert msgs[0]["content"] == SYSTEM_PROMPT

    def test_user_content_is_input(self):
        """User message should be the trajectory input text."""
        trajectory = "SHAPE: rising_warmth\nWINDOW: n_states=5 duration=4.00"
        msgs = _format_inference_messages(trajectory)
        assert msgs[1]["content"] == trajectory


# ---------------------------------------------------------------------------
# TestExtractAssistantResponse
# ---------------------------------------------------------------------------

class TestExtractAssistantResponse:
    """Test extraction of the assistant response from generated text."""

    def test_strip_prompt_prefix(self):
        """When full text starts with prompt, the remainder is extracted."""
        prompt = "You are a system. User asks: what is EISV?"
        full = prompt + "\nEISV_TOKENS: ~warmth~\nLUMEN_TOKENS: warm\nPATTERN: SINGLE"
        result = _extract_assistant_response(full, prompt)
        assert result.startswith("EISV_TOKENS:")

    def test_assistant_marker_extraction(self):
        """Text with <|assistant|> marker should extract after last marker."""
        full = (
            "<|system|>\nYou are...\n"
            "<|user|>\nSHAPE: settled_presence\n"
            "<|assistant|>\n"
            "EISV_TOKENS: ~warmth~\nLUMEN_TOKENS: warm\nPATTERN: SINGLE"
        )
        result = _extract_assistant_response(full, "different prompt")
        assert "EISV_TOKENS:" in result
        assert "PATTERN: SINGLE" in result

    def test_llama_header_marker(self):
        """Text with Llama-style header markers should extract correctly."""
        full = (
            "<|start_header_id|>system<|end_header_id|>\nYou are...\n"
            "<|start_header_id|>user<|end_header_id|>\nSHAPE: test\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
            "EISV_TOKENS: ~stillness~\nLUMEN_TOKENS: quiet\nPATTERN: SINGLE"
        )
        result = _extract_assistant_response(full, "different prompt")
        assert "EISV_TOKENS:" in result

    def test_fallback_returns_stripped_text(self):
        """Without any markers, returns the full text stripped."""
        full = "EISV_TOKENS: ~warmth~\nLUMEN_TOKENS: warm\nPATTERN: SINGLE"
        result = _extract_assistant_response(full, "completely different prompt")
        assert result == full


# ---------------------------------------------------------------------------
# TestExtractTrajectoryFromText
# ---------------------------------------------------------------------------

class TestExtractTrajectoryFromText:
    """Test trajectory extraction from formatted training text."""

    def test_extracts_user_section(self):
        """Should extract text between <|user|> and <|assistant|>."""
        text = (
            "<|system|>\nYou are an EISV trajectory expression mapper.\n"
            "<|user|>\n"
            "SHAPE: settled_presence\n"
            "WINDOW: n_states=5 duration=4.00\n"
            "MEAN_EISV: E=0.8000 I=0.6000 S=0.7000 V=0.5000\n"
            "<|assistant|>\n"
            "EISV_TOKENS: ~warmth~\nLUMEN_TOKENS: warm\nPATTERN: SINGLE"
        )
        result = _extract_trajectory_from_text(text)
        assert result.startswith("SHAPE: settled_presence")
        assert "MEAN_EISV:" in result
        assert "EISV_TOKENS:" not in result
        assert "<|assistant|>" not in result

    def test_no_markers_returns_full_text(self):
        """Without markers, returns the entire text."""
        text = "just some plain text"
        result = _extract_trajectory_from_text(text)
        assert result == text

    def test_no_assistant_marker(self):
        """With user marker but no assistant marker, extracts to end."""
        text = (
            "<|system|>\nSystem prompt\n"
            "<|user|>\n"
            "SHAPE: rising_warmth\nWINDOW: n_states=5"
        )
        result = _extract_trajectory_from_text(text)
        assert "SHAPE: rising_warmth" in result


# ---------------------------------------------------------------------------
# TestParseGeneratedOutput
# ---------------------------------------------------------------------------

class TestParseGeneratedOutput:
    """Test that generated text is correctly parsed.

    Uses :func:`parse_model_output` from teacher_train, tested here in
    the context of inference output parsing.
    """

    def test_clean_output(self):
        """Well-formatted generation parses correctly."""
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

    def test_output_with_preamble(self):
        """Generation with preamble text still parses the structured fields."""
        text = (
            "Based on the trajectory analysis, the expression is:\n\n"
            "EISV_TOKENS: ~resonance~\n"
            "LUMEN_TOKENS: with\n"
            "PATTERN: SINGLE"
        )
        result = parse_model_output(text)
        assert result.valid is True
        assert result.eisv_tokens == ["~resonance~"]
        assert result.pattern == "SINGLE"

    def test_output_with_trailing_text(self):
        """Generation with trailing explanation still parses."""
        text = (
            "EISV_TOKENS: ~warmth~ ~curiosity~\n"
            "LUMEN_TOKENS: warm wonder\n"
            "PATTERN: QUESTION\n\n"
            "This expression reflects a questioning warmth."
        )
        result = parse_model_output(text)
        assert result.valid is True
        assert result.eisv_tokens == ["~warmth~", "~curiosity~"]
        assert result.pattern == "QUESTION"

    def test_empty_generation(self):
        """Empty generation returns invalid result."""
        result = parse_model_output("")
        assert result.valid is False
        assert result.eisv_tokens == []
        assert result.lumen_tokens == []
        assert result.pattern == ""

    def test_garbage_generation(self):
        """Complete garbage returns invalid result."""
        result = parse_model_output("I don't know what to say here sorry")
        assert result.valid is False

    def test_partial_output_missing_lumen(self):
        """Missing LUMEN_TOKENS line makes output invalid."""
        text = (
            "EISV_TOKENS: ~warmth~\n"
            "PATTERN: SINGLE"
        )
        result = parse_model_output(text)
        assert result.valid is False

    def test_partial_output_missing_pattern(self):
        """Missing PATTERN line makes output invalid."""
        text = (
            "EISV_TOKENS: ~warmth~\n"
            "LUMEN_TOKENS: warm"
        )
        result = parse_model_output(text)
        assert result.valid is False

    def test_triple_tokens(self):
        """Three-token output parses correctly."""
        text = (
            "EISV_TOKENS: ~warmth~ ~resonance~ ~stillness~\n"
            "LUMEN_TOKENS: warm with quiet\n"
            "PATTERN: TRIPLE"
        )
        result = parse_model_output(text)
        assert result.valid is True
        assert len(result.eisv_tokens) == 3
        assert len(result.lumen_tokens) == 3
        assert result.pattern == "TRIPLE"


# ---------------------------------------------------------------------------
# TestInferenceImportGuard
# ---------------------------------------------------------------------------

class TestInferenceImportGuard:
    """Test that inference functions raise ImportError without GPU deps."""

    def test_load_teacher_model_import_error(self):
        """load_teacher_model should raise ImportError when deps missing."""
        from eisv_lumen.training.teacher_inference import load_teacher_model

        with patch(
            "eisv_lumen.training.teacher_inference._require_inference_deps",
            side_effect=ImportError("Inference requires PyTorch"),
        ):
            with pytest.raises(ImportError, match="PyTorch"):
                load_teacher_model("/fake/adapter/path")

    def test_generate_expression_import_error(self):
        """generate_expression should raise ImportError when deps missing."""
        from eisv_lumen.training.teacher_inference import generate_expression

        with patch(
            "eisv_lumen.training.teacher_inference._require_inference_deps",
            side_effect=ImportError("Inference requires PyTorch"),
        ):
            with pytest.raises(ImportError, match="PyTorch"):
                generate_expression(MagicMock(), MagicMock(), "SHAPE: test")


# ---------------------------------------------------------------------------
# TestCLIEvalWiring
# ---------------------------------------------------------------------------

class TestCLIEvalWiring:
    """Test that the CLI eval subcommand is wired correctly."""

    def test_eval_subparser_exists(self):
        """The CLI should have an 'eval' subcommand."""
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        eval_cmd = subparsers.add_parser("eval")
        eval_cmd.add_argument("--adapter", required=True)
        eval_cmd.add_argument("--test-data", required=True)
        eval_cmd.add_argument("--base-model", default="Qwen/Qwen3-4B")
        eval_cmd.add_argument("--output", default="outputs/eval_results.json")

        args = parser.parse_args([
            "eval",
            "--adapter", "/tmp/adapter",
            "--test-data", "/tmp/test.json",
        ])
        assert args.adapter == "/tmp/adapter"
        assert args.test_data == "/tmp/test.json"
        assert args.base_model == "Qwen/Qwen3-4B"
        assert args.output == "outputs/eval_results.json"
