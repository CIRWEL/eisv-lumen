"""Training data formatter for EISV trajectory-expression pairs.

Converts trajectory windows and generated expressions into structured
training examples suitable for fine-tuning language models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from eisv_lumen.shapes.expression_generator import ExpressionGenerator
from eisv_lumen.bridge.lumen_bridge import translate_expression


@dataclass
class TrainingExample:
    """A single training example for the teacher model."""

    shape: str
    eisv_tokens: List[str]
    lumen_tokens: List[str]
    pattern: str
    input_text: str
    output_text: str


def _mean(values: List[float]) -> float:
    """Compute mean of a list, returning 0.0 for empty lists."""
    return sum(values) / len(values) if values else 0.0


def format_trajectory_input(shape: str, window: Dict[str, Any]) -> str:
    """Serialize a trajectory window to a text description for model input.

    Parameters
    ----------
    shape:
        Trajectory shape name (e.g. ``"settled_presence"``).
    window:
        Trajectory window dict with ``states``, ``derivatives``,
        ``second_derivatives`` keys.

    Returns
    -------
    Formatted text string containing shape, window metadata, mean EISV
    values, mean derivatives, and mean second derivatives.
    """
    states = window["states"]
    derivs = window.get("derivatives", [])
    second = window.get("second_derivatives", [])

    n_states = len(states)
    duration = states[-1]["t"] - states[0]["t"] if n_states > 1 else 0.0

    # Mean EISV values
    mean_e = _mean([s["E"] for s in states])
    mean_i = _mean([s["I"] for s in states])
    mean_s = _mean([s["S"] for s in states])
    mean_v = _mean([s["V"] for s in states])

    # Mean first derivatives
    mean_de = _mean([d["dE"] for d in derivs]) if derivs else 0.0
    mean_di = _mean([d["dI"] for d in derivs]) if derivs else 0.0
    mean_ds = _mean([d["dS"] for d in derivs]) if derivs else 0.0
    mean_dv = _mean([d["dV"] for d in derivs]) if derivs else 0.0

    # Mean second derivatives
    mean_d2e = _mean([d["d2E"] for d in second]) if second else 0.0
    mean_d2i = _mean([d["d2I"] for d in second]) if second else 0.0
    mean_d2s = _mean([d["d2S"] for d in second]) if second else 0.0
    mean_d2v = _mean([d["d2V"] for d in second]) if second else 0.0

    lines = [
        f"SHAPE: {shape}",
        f"WINDOW: n_states={n_states} duration={duration:.2f}",
        f"MEAN_EISV: E={mean_e:.4f} I={mean_i:.4f} S={mean_s:.4f} V={mean_v:.4f}",
        f"DERIVATIVES: dE={mean_de:.4f} dI={mean_di:.4f} dS={mean_ds:.4f} dV={mean_dv:.4f}",
        f"SECOND_DERIVATIVES: d2E={mean_d2e:.4f} d2I={mean_d2i:.4f} d2S={mean_d2s:.4f} d2V={mean_d2v:.4f}",
    ]
    return "\n".join(lines)


def format_expression_output(
    eisv_tokens: List[str],
    lumen_tokens: List[str],
    pattern: str,
) -> str:
    """Format expression data as the target output for training.

    Parameters
    ----------
    eisv_tokens:
        List of EISV-Lumen expression tokens.
    lumen_tokens:
        List of translated Lumen primitive tokens.
    pattern:
        Expression pattern name (e.g. ``"SINGLE"``, ``"PAIR"``).

    Returns
    -------
    Formatted string with EISV_TOKENS, LUMEN_TOKENS, and PATTERN lines.
    """
    lines = [
        f"EISV_TOKENS: {' '.join(eisv_tokens)}",
        f"LUMEN_TOKENS: {' '.join(lumen_tokens)}",
        f"PATTERN: {pattern}",
    ]
    return "\n".join(lines)


def _infer_pattern(tokens: List[str]) -> str:
    """Infer expression pattern from a token list.

    Parameters
    ----------
    tokens:
        List of EISV-Lumen tokens.

    Returns
    -------
    One of ``"SINGLE"``, ``"PAIR"``, ``"TRIPLE"``, ``"REPETITION"``,
    ``"QUESTION"``.
    """
    inquiry_tokens = {"~questioning~", "~curiosity~"}

    if len(tokens) == 1:
        return "SINGLE"

    # Check for repetition (any token appears more than once)
    if len(tokens) == 2 and tokens[0] == tokens[1]:
        return "REPETITION"

    # Check for question (ends with an inquiry token)
    if tokens[-1] in inquiry_tokens:
        return "QUESTION"

    if len(tokens) == 2:
        return "PAIR"

    if len(tokens) >= 3:
        return "TRIPLE"

    return "SINGLE"


def build_training_example(
    shape: str,
    window: Dict[str, Any],
    seed: Optional[int] = None,
) -> TrainingExample:
    """Build a complete training example from a trajectory shape and window.

    Uses :class:`ExpressionGenerator` to produce EISV tokens, translates
    them via the Lumen bridge, infers the pattern, and formats input/output
    text.

    Parameters
    ----------
    shape:
        Trajectory shape name.
    window:
        Trajectory window dict with ``states``, ``derivatives``,
        ``second_derivatives``.
    seed:
        Random seed for deterministic expression generation.

    Returns
    -------
    A fully populated :class:`TrainingExample`.
    """
    gen = ExpressionGenerator(seed=seed)
    eisv_tokens = gen.generate(shape)
    lumen_tokens = translate_expression(eisv_tokens)
    pattern = _infer_pattern(eisv_tokens)

    input_text = format_trajectory_input(shape, window)
    output_text = format_expression_output(eisv_tokens, lumen_tokens, pattern)

    return TrainingExample(
        shape=shape,
        eisv_tokens=eisv_tokens,
        lumen_tokens=lumen_tokens,
        pattern=pattern,
        input_text=input_text,
        output_text=output_text,
    )
