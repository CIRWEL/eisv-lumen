"""Chat template formatter for teacher model training.

Converts TrainingExample instances into chat-formatted message lists
suitable for instruction-tuned LLM fine-tuning.
"""

from __future__ import annotations

from typing import Any, Dict, List

from eisv_lumen.training.data_prep import TrainingExample


SYSTEM_PROMPT = (
    "You are an EISV trajectory expression mapper. Given a trajectory window "
    "with its shape classification, EISV state values, derivatives, and second "
    "derivatives, generate an appropriate primitive expression. Output the "
    "EISV tokens, their Lumen translations, and the expression pattern."
)


def format_chat_messages(example: TrainingExample) -> List[Dict[str, str]]:
    """Convert a TrainingExample into a chat message list.

    Produces a three-message conversation:
    1. System message with the instruction prompt
    2. User message with the trajectory input
    3. Assistant message with the expression output

    Parameters
    ----------
    example:
        A :class:`TrainingExample` instance.

    Returns
    -------
    List of dicts with ``role`` and ``content`` keys, suitable for
    chat-formatted model training.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example.input_text},
        {"role": "assistant", "content": example.output_text},
    ]


def format_for_tokenizer(example: TrainingExample) -> Dict[str, Any]:
    """Format a TrainingExample for tokenizer consumption.

    Returns a dict containing the full chat text, message list, and
    metadata fields for filtering and analysis.

    Parameters
    ----------
    example:
        A :class:`TrainingExample` instance.

    Returns
    -------
    Dict with keys:
        - ``text``: formatted chat conversation as a single string
        - ``shape``: trajectory shape name
        - ``pattern``: expression pattern name
        - ``messages``: list of message dicts (from :func:`format_chat_messages`)
    """
    messages = format_chat_messages(example)

    # Build a simple text representation of the chat
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|{role}|>\n{content}")
    text = "\n".join(parts)

    return {
        "text": text,
        "shape": example.shape,
        "pattern": example.pattern,
        "messages": messages,
    }
