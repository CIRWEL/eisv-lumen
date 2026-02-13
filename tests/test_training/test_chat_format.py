"""Tests for eisv_lumen.training.chat_format — chat template formatter."""

from __future__ import annotations

import pytest

from eisv_lumen.training.data_prep import TrainingExample
from eisv_lumen.training.chat_format import (
    SYSTEM_PROMPT,
    format_chat_messages,
    format_for_tokenizer,
)


def _make_example() -> TrainingExample:
    """Create a sample TrainingExample for testing."""
    return TrainingExample(
        shape="settled_presence",
        eisv_tokens=["~stillness~", "~holding~"],
        lumen_tokens=["quiet", "here"],
        pattern="PAIR",
        input_text="SHAPE: settled_presence\nWINDOW: n_states=20 duration=19.00\nMEAN_EISV: E=0.7000 I=0.6000 S=0.1500 V=0.0500\nDERIVATIVES: dE=0.0000 dI=0.0000 dS=0.0000 dV=0.0000\nSECOND_DERIVATIVES: d2E=0.0000 d2I=0.0000 d2S=0.0000 d2V=0.0000",
        output_text="EISV_TOKENS: ~stillness~ ~holding~\nLUMEN_TOKENS: quiet here\nPATTERN: PAIR",
    )


# ---------------------------------------------------------------------------
# TestFormatChatMessages
# ---------------------------------------------------------------------------


class TestFormatChatMessages:
    def test_returns_message_list(self):
        """Should return a list of 3 message dicts."""
        messages = format_chat_messages(_make_example())
        assert isinstance(messages, list)
        assert len(messages) == 3
        for msg in messages:
            assert "role" in msg
            assert "content" in msg

    def test_system_prompt_present(self):
        """First message should be the system prompt."""
        messages = format_chat_messages(_make_example())
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT
        assert len(SYSTEM_PROMPT) > 20  # Non-trivial system prompt

    def test_user_contains_input(self):
        """User message should contain the trajectory input text."""
        example = _make_example()
        messages = format_chat_messages(example)
        user_msg = messages[1]
        assert user_msg["role"] == "user"
        assert example.input_text in user_msg["content"]

    def test_assistant_contains_output(self):
        """Assistant message should contain the expression output text."""
        example = _make_example()
        messages = format_chat_messages(example)
        assistant_msg = messages[2]
        assert assistant_msg["role"] == "assistant"
        assert example.output_text in assistant_msg["content"]


# ---------------------------------------------------------------------------
# TestFormatForTokenizer
# ---------------------------------------------------------------------------


class TestFormatForTokenizer:
    def test_returns_dict_with_text(self):
        """Should return a dict with 'text', 'shape', 'pattern', 'messages' keys."""
        result = format_for_tokenizer(_make_example())
        assert isinstance(result, dict)
        assert "text" in result
        assert "messages" in result
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0

    def test_metadata_preserved(self):
        """Shape and pattern should be preserved in output metadata."""
        example = _make_example()
        result = format_for_tokenizer(example)
        assert result["shape"] == "settled_presence"
        assert result["pattern"] == "PAIR"
