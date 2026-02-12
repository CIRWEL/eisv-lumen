"""Baseline evaluation conditions for EISV trajectory-expression benchmarking.

Provides three baseline generation strategies and an evaluation runner
for comparing dynamics-emergent voice output against simple baselines:

* RANDOM – uniform random token selection (lower bound)
* SHAPE_MATCHED – oracle that always picks affine tokens (upper bound)
* PROMPT_CONDITIONED – simulated LLM with 70/30 affine/random mix
"""

from __future__ import annotations

import random
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List

from eisv_lumen.eval.metrics import (
    SHAPE_TOKEN_AFFINITY,
    expression_trajectory_coherence,
    vocabulary_diversity_per_shape,
)
from eisv_lumen.shapes.shape_classes import TrajectoryShape


ALL_TOKENS: List[str] = [
    "~warmth~",
    "~curiosity~",
    "~resonance~",
    "~stillness~",
    "~boundary~",
    "~reaching~",
    "~reflection~",
    "~ripple~",
    "~deep_listening~",
    "~emergence~",
    "~questioning~",
    "~holding~",
    "~releasing~",
    "~threshold~",
    "~return~",
]


class BaselineCondition(str, Enum):
    """Baseline token-generation strategies."""

    RANDOM = "random"
    SHAPE_MATCHED = "shape_matched"
    PROMPT_CONDITIONED = "prompt_conditioned"


def generate_baseline_tokens(
    shape: str,
    condition: BaselineCondition,
    n_tokens: int = 3,
    rng: random.Random | None = None,
) -> List[str]:
    """Generate tokens under a baseline condition for a given shape.

    Parameters
    ----------
    shape:
        Trajectory shape name (e.g. ``"settled_presence"``).
    condition:
        Which baseline strategy to use.
    n_tokens:
        How many tokens to generate.
    rng:
        A :class:`random.Random` instance for reproducibility.
        If *None*, a default instance is created.

    Returns
    -------
    List of primitive token strings.

    Notes
    -----
    Falls back to RANDOM selection if *shape* is not present in the
    affinity matrix (``SHAPE_TOKEN_AFFINITY``).
    """
    if rng is None:
        rng = random.Random()

    affinity = SHAPE_TOKEN_AFFINITY.get(shape)

    # If shape unknown, force RANDOM regardless of requested condition
    if affinity is None:
        return [rng.choice(ALL_TOKENS) for _ in range(n_tokens)]

    if condition is BaselineCondition.RANDOM:
        return [rng.choice(ALL_TOKENS) for _ in range(n_tokens)]

    if condition is BaselineCondition.SHAPE_MATCHED:
        return [rng.choice(affinity) for _ in range(n_tokens)]

    # PROMPT_CONDITIONED: 70 % affine, 30 % uniform random
    tokens: List[str] = []
    for _ in range(n_tokens):
        if rng.random() < 0.7:
            tokens.append(rng.choice(affinity))
        else:
            tokens.append(rng.choice(ALL_TOKENS))
    return tokens


def evaluate_baseline(
    trajectory_records: List[Dict[str, Any]],
    condition: BaselineCondition,
    n_tokens: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a baseline condition on trajectory records and compute metrics.

    Parameters
    ----------
    trajectory_records:
        List of dicts with at least a ``"shape"`` key.
    condition:
        Baseline strategy to apply.
    n_tokens:
        Number of tokens to generate per record.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Dict with keys:
        - ``condition`` – the condition name string
        - ``mean_coherence`` – mean expression–trajectory coherence
        - ``per_shape_coherence`` – dict mapping shape -> mean coherence
        - ``diversity`` – output of :func:`vocabulary_diversity_per_shape`
        - ``n_records`` – number of input records
    """
    if not trajectory_records:
        return {
            "condition": condition.value,
            "mean_coherence": 0.0,
            "per_shape_coherence": {},
            "diversity": {},
            "n_records": 0,
        }

    rng = random.Random(seed)

    # Generate tokens for each record and compute per-record coherence
    augmented_records: List[Dict[str, Any]] = []
    coherence_scores: List[float] = []
    per_shape_scores: Dict[str, List[float]] = defaultdict(list)

    for record in trajectory_records:
        shape = record["shape"]
        tokens = generate_baseline_tokens(shape, condition, n_tokens=n_tokens, rng=rng)
        score = expression_trajectory_coherence(shape, tokens)

        coherence_scores.append(score)
        per_shape_scores[shape].append(score)
        augmented_records.append({"shape": shape, "tokens": tokens})

    mean_coherence = sum(coherence_scores) / len(coherence_scores)

    per_shape_coherence = {
        shape: sum(scores) / len(scores)
        for shape, scores in per_shape_scores.items()
    }

    diversity = vocabulary_diversity_per_shape(augmented_records)

    return {
        "condition": condition.value,
        "mean_coherence": mean_coherence,
        "per_shape_coherence": per_shape_coherence,
        "diversity": diversity,
        "n_records": len(trajectory_records),
    }


def format_eisv_prompt(
    eisv_state: Dict[str, float],
    shape: str,
) -> str:
    """Format an EISV state and shape into a prompt string for LLM comparison.

    Parameters
    ----------
    eisv_state:
        Dict with keys ``"E"``, ``"I"``, ``"S"``, ``"V"`` mapping to floats.
    shape:
        Trajectory shape name.

    Returns
    -------
    A formatted prompt string.
    """
    e = eisv_state.get("E", 0.0)
    i = eisv_state.get("I", 0.0)
    s = eisv_state.get("S", 0.0)
    v = eisv_state.get("V", 0.0)

    return (
        f"EISV State: E={e} I={i} S={s} V={v}\n"
        f"Trajectory: {shape}\n"
        f"Express this state using primitive tokens."
    )
