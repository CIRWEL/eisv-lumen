"""Evaluation metrics for trajectory-expression coherence.

Provides three core metrics for benchmarking the dynamics-emergent
voice system against the EISV governance framework:

* shape_classification_accuracy – simple label-match accuracy
* expression_trajectory_coherence – SHAPE_TOKEN_AFFINITY-based score
* vocabulary_diversity_per_shape – unique-token ratio per shape class
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Affinity matrix: which primitive tokens are coherent for each shape class
# ---------------------------------------------------------------------------

SHAPE_TOKEN_AFFINITY: Dict[str, List[str]] = {
    "settled_presence": ["~stillness~", "~holding~", "~resonance~", "~deep_listening~"],
    "rising_entropy": ["~ripple~", "~emergence~", "~questioning~", "~curiosity~"],
    "falling_energy": ["~releasing~", "~stillness~", "~boundary~", "~reflection~"],
    "basin_transition_down": ["~releasing~", "~threshold~", "~boundary~"],
    "basin_transition_up": ["~emergence~", "~reaching~", "~warmth~", "~return~"],
    "entropy_spike_recovery": ["~ripple~", "~return~", "~holding~", "~reflection~"],
    "drift_dissonance": ["~boundary~", "~questioning~", "~reflection~"],
    "void_rising": ["~reaching~", "~curiosity~", "~questioning~", "~threshold~"],
    "convergence": ["~stillness~", "~resonance~", "~return~", "~deep_listening~"],
}


# ---------------------------------------------------------------------------
# Metric 1 – shape classification accuracy
# ---------------------------------------------------------------------------

def shape_classification_accuracy(
    predicted: List[str],
    actual: List[str],
) -> float:
    """Simple accuracy: fraction of predictions matching actual labels.

    Returns 0.0 if lists are empty.
    """
    if not predicted:
        return 0.0
    matches = sum(1 for p, a in zip(predicted, actual) if p == a)
    return matches / len(predicted)


# ---------------------------------------------------------------------------
# Metric 2 – expression–trajectory coherence
# ---------------------------------------------------------------------------

def expression_trajectory_coherence(
    shape: str,
    tokens: List[str],
) -> float:
    """Score how coherent a set of tokens is for a given shape class.

    Uses :data:`SHAPE_TOKEN_AFFINITY` matrix.
    Score = (number of tokens in affinity list) / (total tokens).
    Returns 0.0 if *tokens* is empty or *shape* not in affinity matrix.
    """
    if not tokens or shape not in SHAPE_TOKEN_AFFINITY:
        return 0.0
    affinity_set = set(SHAPE_TOKEN_AFFINITY[shape])
    hits = sum(1 for t in tokens if t in affinity_set)
    return hits / len(tokens)


# ---------------------------------------------------------------------------
# Metric 3 – vocabulary diversity per shape
# ---------------------------------------------------------------------------

def vocabulary_diversity_per_shape(
    records: List[Dict[str, Any]],
) -> Dict[str, float]:
    """For each shape class present in *records*, compute vocabulary diversity.

    Parameters
    ----------
    records:
        List of dicts with ``"shape"`` (str) and ``"tokens"`` (List[str]) keys.

    Returns
    -------
    Dict mapping shape -> diversity float.
    Diversity = unique tokens / total tokens for that shape.
    Shapes with 0 total tokens get diversity 0.0.
    """
    all_tokens: Dict[str, List[str]] = defaultdict(list)
    for rec in records:
        all_tokens[rec["shape"]].extend(rec["tokens"])

    result: Dict[str, float] = {}
    for shape, tokens in all_tokens.items():
        if not tokens:
            result[shape] = 0.0
        else:
            result[shape] = len(set(tokens)) / len(tokens)
    return result
