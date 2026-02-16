"""Teacher training pipeline for EISV trajectory-expression mapping.

Provides functions for preparing training data, parsing model outputs,
and validating generated expressions against the known EISV and Lumen
token vocabularies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from eisv_lumen.eval.baseline import ALL_TOKENS
from eisv_lumen.bridge.lumen_bridge import LUMEN_TOKENS
from eisv_lumen.training.dataset_builder import build_training_dataset, split_dataset
from eisv_lumen.training.chat_format import format_for_tokenizer

VALID_PATTERNS = {"SINGLE", "PAIR", "TRIPLE", "REPETITION", "QUESTION"}

EISV_TOKEN_SET = set(ALL_TOKENS)
LUMEN_TOKEN_SET = set(LUMEN_TOKENS)


@dataclass
class OutputParseResult:
    """Result of parsing a model's text output into structured fields."""

    eisv_tokens: List[str] = field(default_factory=list)
    lumen_tokens: List[str] = field(default_factory=list)
    pattern: str = ""
    valid: bool = False


def prepare_training_data(
    real_records: List[Dict],
    min_per_shape: int = 50,
    seed: int = 42,
    shape_overrides: Optional[Dict[str, int]] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Build and split a training dataset, formatted for tokenizer consumption.

    Uses :func:`build_training_dataset` to create balanced examples,
    :func:`split_dataset` to partition into train/val/test, then
    :func:`format_for_tokenizer` on each example.

    Parameters
    ----------
    real_records:
        List of real trajectory record dicts (may be empty for synthetic-only).
    min_per_shape:
        Minimum examples per shape class.
    seed:
        Random seed for reproducibility.
    shape_overrides:
        Optional dict mapping shape names to per-shape minimums.

    Returns
    -------
    Tuple of (train, val, test) where each element is a list of dicts
    with keys ``text``, ``shape``, ``pattern``, ``messages``.
    """
    examples = build_training_dataset(
        real_records, min_per_shape=min_per_shape, seed=seed,
        shape_overrides=shape_overrides,
    )
    train_ex, val_ex, test_ex = split_dataset(examples, seed=seed)

    train = [format_for_tokenizer(ex) for ex in train_ex]
    val = [format_for_tokenizer(ex) for ex in val_ex]
    test = [format_for_tokenizer(ex) for ex in test_ex]

    return train, val, test


def parse_model_output(text: str) -> OutputParseResult:
    """Parse structured fields from a model's text output.

    Expects lines of the form::

        EISV_TOKENS: ~warmth~ ~stillness~
        LUMEN_TOKENS: warm quiet
        PATTERN: PAIR

    Parameters
    ----------
    text:
        Raw model output text.

    Returns
    -------
    An :class:`OutputParseResult`. ``valid`` is True only if all three
    fields (eisv_tokens, lumen_tokens, pattern) are non-empty.
    """
    eisv_tokens: List[str] = []
    lumen_tokens: List[str] = []
    pattern: str = ""

    eisv_match = re.search(r"EISV_TOKENS:\s*(.+)", text)
    if eisv_match:
        eisv_tokens = eisv_match.group(1).strip().split()

    lumen_match = re.search(r"LUMEN_TOKENS:\s*(.+)", text)
    if lumen_match:
        lumen_tokens = lumen_match.group(1).strip().split()

    pattern_match = re.search(r"PATTERN:\s*(\S+)", text)
    if pattern_match:
        pattern = pattern_match.group(1).strip()

    valid = bool(eisv_tokens and lumen_tokens and pattern)

    return OutputParseResult(
        eisv_tokens=eisv_tokens,
        lumen_tokens=lumen_tokens,
        pattern=pattern,
        valid=valid,
    )


def validate_output(result: OutputParseResult) -> bool:
    """Validate a parsed output against known token vocabularies and patterns.

    Checks that:
    - ``result.valid`` is True (all fields were parsed)
    - All EISV tokens are in :data:`EISV_TOKEN_SET`
    - All Lumen tokens are in :data:`LUMEN_TOKEN_SET`
    - Pattern is in :data:`VALID_PATTERNS`

    Parameters
    ----------
    result:
        A parsed :class:`OutputParseResult`.

    Returns
    -------
    True if the output is fully valid, False otherwise.
    """
    if not result.valid:
        return False

    if not all(t in EISV_TOKEN_SET for t in result.eisv_tokens):
        return False

    if not all(t in LUMEN_TOKEN_SET for t in result.lumen_tokens):
        return False

    if result.pattern not in VALID_PATTERNS:
        return False

    return True
