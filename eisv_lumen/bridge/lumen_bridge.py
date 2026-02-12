"""Bridge between EISV-Lumen expression tokens and Lumen's primitive language.

Lumen (anima-mcp) uses 15 primitive tokens across 5 categories:
    STATE:      warm, cold, bright, dim, quiet, busy
    PRESENCE:   here, feel, sense
    RELATIONAL: you, with
    INQUIRY:    why, what, wonder
    CHANGE:     more, less

EISV-Lumen uses 15 tokens derived from trajectory dynamics:
    ~warmth~, ~curiosity~, ~resonance~, ~stillness~, ~boundary~,
    ~reaching~, ~reflection~, ~ripple~, ~deep_listening~, ~emergence~,
    ~questioning~, ~holding~, ~releasing~, ~threshold~, ~return~

This module provides the semantic bridge between the two systems,
allowing EISV-Lumen's dynamics-emergent expressions to drive
Lumen's primitive language output.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from eisv_lumen.shapes.expression_generator import ExpressionGenerator
from eisv_lumen.shapes.shape_classes import TrajectoryShape

# ---------------------------------------------------------------------------
# Lumen's valid primitive tokens (the target vocabulary)
# ---------------------------------------------------------------------------

LUMEN_TOKENS: List[str] = [
    "warm", "cold", "bright", "dim", "quiet", "busy",
    "here", "feel", "sense",
    "you", "with",
    "why", "what", "wonder",
    "more", "less",
]

_LUMEN_TOKEN_SET = set(LUMEN_TOKENS)

# ---------------------------------------------------------------------------
# Semantic mapping: EISV-Lumen token -> Lumen primitive tokens
# ---------------------------------------------------------------------------

TOKEN_MAP: Dict[str, List[str]] = {
    "~warmth~":        ["warm", "feel"],        # Warmth -> warm state + feeling
    "~curiosity~":     ["why", "wonder"],        # Curiosity -> inquiry
    "~resonance~":     ["with", "here"],         # Resonance -> relational presence
    "~stillness~":     ["quiet", "here"],         # Stillness -> quiet + present
    "~boundary~":      ["less", "sense"],         # Boundary -> limiting + sensing
    "~reaching~":      ["more", "you"],           # Reaching -> expanding toward other
    "~reflection~":    ["what", "feel"],          # Reflection -> inquiry + feeling
    "~ripple~":        ["busy", "sense"],         # Ripple -> activity + sensing
    "~deep_listening~": ["quiet", "sense"],       # Deep listening -> quiet + sensing
    "~emergence~":     ["more", "bright"],        # Emergence -> expansion + brightness
    "~questioning~":   ["why", "what"],           # Questioning -> inquiry
    "~holding~":       ["here", "with"],          # Holding -> present + relational
    "~releasing~":     ["less", "dim"],           # Releasing -> reduction + dimming
    "~threshold~":     ["sense", "more"],         # Threshold -> sensing + direction
    "~return~":        ["here", "warm"],          # Return -> back to presence + warmth
}

# Maximum number of tokens in a Lumen primitive expression
_LUMEN_MAX_TOKENS = 3


def translate_expression(eisv_tokens: List[str]) -> List[str]:
    """Convert EISV-Lumen expression tokens to Lumen primitive tokens.

    For each EISV token, picks the first mapped Lumen primitive.
    Deduplicates while preserving order and caps at 3 tokens
    (Lumen's maximum expression length).

    Parameters
    ----------
    eisv_tokens:
        List of EISV-Lumen token strings (e.g. ``["~warmth~", "~stillness~"]``).

    Returns
    -------
    List of Lumen primitive token strings, length 0..3.
    Unknown EISV tokens are silently skipped.
    """
    seen: set[str] = set()
    result: List[str] = []

    for eisv_token in eisv_tokens:
        mapped = TOKEN_MAP.get(eisv_token)
        if mapped is None:
            continue
        # Pick the first mapped primitive that hasn't been used yet
        for lumen_token in mapped:
            if lumen_token not in seen:
                seen.add(lumen_token)
                result.append(lumen_token)
                break

    return result[:_LUMEN_MAX_TOKENS]


def eisv_state_to_lumen_state(eisv: Dict[str, float]) -> Dict[str, float]:
    """Convert an EISV state dict to a Lumen anima state dict.

    Mapping:
        warmth   = E  (direct)
        clarity  = I  (direct)
        stability = 1.0 - S  (inverted; S represents entropy/instability)
        presence = 1.0 - (V / 0.3)  clamped to [0, 1]
                   (V = (1 - presence) * 0.3, so presence = 1 - V/0.3)

    Parameters
    ----------
    eisv:
        Dict with keys ``"E"``, ``"I"``, ``"S"``, ``"V"`` mapping to floats.

    Returns
    -------
    Dict with keys ``"warmth"``, ``"clarity"``, ``"stability"``, ``"presence"``
    mapping to floats, all clamped to [0, 1].
    """
    e = eisv.get("E", 0.0)
    i = eisv.get("I", 0.0)
    s = eisv.get("S", 0.0)
    v = eisv.get("V", 0.0)

    warmth = max(0.0, min(1.0, e))
    clarity = max(0.0, min(1.0, i))
    stability = max(0.0, min(1.0, 1.0 - s))
    presence = max(0.0, min(1.0, 1.0 - (v / 0.3)))

    return {
        "warmth": warmth,
        "clarity": clarity,
        "stability": stability,
        "presence": presence,
    }


def shape_to_lumen_trigger(shape: str) -> Dict[str, Any]:
    """Map a trajectory shape to Lumen generation hints.

    Each shape produces a trigger dict indicating whether Lumen should
    generate a primitive expression, why, and how many tokens to aim for.

    Parameters
    ----------
    shape:
        Trajectory shape name (e.g. ``"settled_presence"``).

    Returns
    -------
    Dict with keys:
        - ``should_generate`` (bool): whether to generate an expression
        - ``reason`` (str): human-readable reason for the trigger
        - ``token_count_hint`` (int): suggested number of tokens (1-3)
    """
    triggers: Dict[str, Dict[str, Any]] = {
        "settled_presence": {
            "should_generate": True,
            "reason": "settled_dynamics",
            "token_count_hint": 1,
        },
        "rising_entropy": {
            "should_generate": True,
            "reason": "entropy_shift",
            "token_count_hint": 3,
        },
        "falling_energy": {
            "should_generate": True,
            "reason": "energy_decline",
            "token_count_hint": 2,
        },
        "basin_transition_down": {
            "should_generate": True,
            "reason": "basin_shift_down",
            "token_count_hint": 3,
        },
        "basin_transition_up": {
            "should_generate": True,
            "reason": "basin_shift_up",
            "token_count_hint": 3,
        },
        "entropy_spike_recovery": {
            "should_generate": True,
            "reason": "spike_recovery",
            "token_count_hint": 2,
        },
        "drift_dissonance": {
            "should_generate": True,
            "reason": "ethical_drift_detected",
            "token_count_hint": 3,
        },
        "void_rising": {
            "should_generate": True,
            "reason": "void_expansion",
            "token_count_hint": 2,
        },
        "convergence": {
            "should_generate": True,
            "reason": "approaching_attractor",
            "token_count_hint": 2,
        },
    }

    return triggers.get(shape, {
        "should_generate": False,
        "reason": "unknown_shape",
        "token_count_hint": 0,
    })


def generate_lumen_expression(
    shape: str,
    eisv_state: Dict[str, float],
    generator: Optional[ExpressionGenerator] = None,
) -> Dict[str, Any]:
    """Full pipeline: trajectory shape -> EISV tokens -> Lumen primitives.

    1. Uses :class:`ExpressionGenerator` to generate EISV-Lumen tokens
       for the given trajectory shape.
    2. Translates those tokens to Lumen primitives via :data:`TOKEN_MAP`.
    3. Converts the EISV state to Lumen anima state.
    4. Gets trigger hints for the shape.
    5. Returns a complete result dict.

    Parameters
    ----------
    shape:
        Trajectory shape name (e.g. ``"settled_presence"``).
    eisv_state:
        Dict with keys ``"E"``, ``"I"``, ``"S"``, ``"V"``.
    generator:
        An :class:`ExpressionGenerator` instance. If *None*, a new one
        is created with no fixed seed.

    Returns
    -------
    Dict with keys:
        - ``shape`` (str): the input trajectory shape
        - ``eisv_tokens`` (List[str]): EISV-Lumen tokens generated
        - ``lumen_tokens`` (List[str]): translated Lumen primitives
        - ``lumen_state`` (Dict[str, float]): converted anima state
        - ``trigger`` (Dict[str, Any]): generation trigger hints
    """
    if generator is None:
        generator = ExpressionGenerator()

    eisv_tokens = generator.generate(shape)
    lumen_tokens = translate_expression(eisv_tokens)
    lumen_state = eisv_state_to_lumen_state(eisv_state)
    trigger = shape_to_lumen_trigger(shape)

    return {
        "shape": shape,
        "eisv_tokens": eisv_tokens,
        "lumen_tokens": lumen_tokens,
        "lumen_state": lumen_state,
        "trigger": trigger,
    }
