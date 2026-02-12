"""Dynamics-emergent expression generator — primary research contribution.

Generates primitive expressions directly from trajectory shape dynamics
using rule-based, interpretable methods. Instead of relying on a neural
network, the system uses:

1. **Shape-driven pattern selection** — each trajectory shape has its own
   probability distribution over expression patterns (SINGLE, PAIR, TRIPLE,
   REPETITION, QUESTION).

2. **Affinity-weighted token sampling** — tokens coherent with the current
   shape receive higher sampling weight (3.0 vs 1.0 baseline), producing
   contextually appropriate expressions.

3. **Feedback-driven weight updates** — an online learning loop adjusts
   token weights based on coherence feedback, allowing the system to adapt
   over time while remaining fully interpretable.
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from eisv_lumen.eval.metrics import SHAPE_TOKEN_AFFINITY
from eisv_lumen.eval.baseline import ALL_TOKENS
from eisv_lumen.shapes.shape_classes import TrajectoryShape


class ExpressionPattern(str, Enum):
    """Structural patterns for primitive expressions."""

    SINGLE = "single"          # One token: "~stillness~"
    PAIR = "pair"              # Two tokens: "~stillness~ ~holding~"
    TRIPLE = "triple"          # Three tokens: "~stillness~ ~holding~ ~resonance~"
    REPETITION = "repetition"  # Repeated token: "~stillness~ ~stillness~"
    QUESTION = "question"      # Ends with inquiry token: "~warmth~ ~questioning~"


# Per-shape pattern distributions (probabilities).
# Each shape maps to a dict of pattern -> probability.
# Probabilities sum to 1.0 for each shape.
SHAPE_PATTERN_WEIGHTS: Dict[str, Dict[str, float]] = {
    "settled_presence":        {"single": 0.4, "pair": 0.3, "triple": 0.1, "repetition": 0.15, "question": 0.05},
    "rising_entropy":          {"single": 0.1, "pair": 0.2, "triple": 0.3, "repetition": 0.1, "question": 0.3},
    "falling_energy":          {"single": 0.3, "pair": 0.3, "triple": 0.1, "repetition": 0.2, "question": 0.1},
    "basin_transition_down":   {"single": 0.2, "pair": 0.3, "triple": 0.3, "repetition": 0.1, "question": 0.1},
    "basin_transition_up":     {"single": 0.15, "pair": 0.3, "triple": 0.35, "repetition": 0.1, "question": 0.1},
    "entropy_spike_recovery":  {"single": 0.1, "pair": 0.3, "triple": 0.3, "repetition": 0.2, "question": 0.1},
    "drift_dissonance":        {"single": 0.1, "pair": 0.2, "triple": 0.2, "repetition": 0.1, "question": 0.4},
    "void_rising":             {"single": 0.2, "pair": 0.2, "triple": 0.2, "repetition": 0.1, "question": 0.3},
    "convergence":             {"single": 0.4, "pair": 0.3, "triple": 0.1, "repetition": 0.15, "question": 0.05},
}

INQUIRY_TOKENS = ["~questioning~", "~curiosity~"]


class ExpressionGenerator:
    """Generate primitive expressions shaped by trajectory dynamics.

    This is the core of the dynamics-emergent voice system:
    trajectory shape -> (pattern selection, token weighting) -> expression

    Parameters
    ----------
    seed:
        Optional random seed for reproducible generation.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        # Token weights per shape: start from affinity, can be updated by feedback
        self._token_weights: Dict[str, Dict[str, float]] = {}
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize token weights from SHAPE_TOKEN_AFFINITY.

        Affine tokens get weight 3.0, non-affine get weight 1.0.
        """
        for shape in TrajectoryShape:
            affine = set(SHAPE_TOKEN_AFFINITY.get(shape.value, []))
            weights: Dict[str, float] = {}
            for token in ALL_TOKENS:
                weights[token] = 3.0 if token in affine else 1.0
            self._token_weights[shape.value] = weights

    def _select_pattern(self, shape: str) -> ExpressionPattern:
        """Select an expression pattern based on shape distribution."""
        weights = SHAPE_PATTERN_WEIGHTS.get(
            shape, SHAPE_PATTERN_WEIGHTS["settled_presence"]
        )
        patterns = list(weights.keys())
        probs = list(weights.values())
        chosen = self.rng.choices(patterns, weights=probs, k=1)[0]
        return ExpressionPattern(chosen)

    def _weighted_token_choice(
        self, shape: str, exclude: Optional[set] = None
    ) -> str:
        """Choose a single token using shape-specific weights.

        Parameters
        ----------
        shape:
            Trajectory shape name.
        exclude:
            Set of tokens to exclude from selection (for diversity).
        """
        weights = self._token_weights.get(
            shape, {t: 1.0 for t in ALL_TOKENS}
        )
        tokens = list(weights.keys())
        w = list(weights.values())
        if exclude:
            filtered = [(t, wt) for t, wt in zip(tokens, w) if t not in exclude]
            if filtered:
                tokens, w = zip(*filtered)
                tokens, w = list(tokens), list(w)
        return self.rng.choices(tokens, weights=w, k=1)[0]

    def generate(self, shape: str) -> List[str]:
        """Generate a primitive expression (list of tokens) for a trajectory shape.

        The pattern and tokens are both shaped by the trajectory class.

        Parameters
        ----------
        shape:
            Trajectory shape name (e.g. ``"settled_presence"``).

        Returns
        -------
        List of primitive token strings forming the expression.
        """
        pattern = self._select_pattern(shape)

        if pattern == ExpressionPattern.SINGLE:
            return [self._weighted_token_choice(shape)]

        elif pattern == ExpressionPattern.PAIR:
            t1 = self._weighted_token_choice(shape)
            t2 = self._weighted_token_choice(shape, exclude={t1})
            return [t1, t2]

        elif pattern == ExpressionPattern.TRIPLE:
            t1 = self._weighted_token_choice(shape)
            t2 = self._weighted_token_choice(shape, exclude={t1})
            t3 = self._weighted_token_choice(shape, exclude={t1, t2})
            return [t1, t2, t3]

        elif pattern == ExpressionPattern.REPETITION:
            t = self._weighted_token_choice(shape)
            return [t, t]

        elif pattern == ExpressionPattern.QUESTION:
            t1 = self._weighted_token_choice(shape)
            t2 = self.rng.choice(INQUIRY_TOKENS)
            return [t1, t2]

        # Fallback (should not be reached)
        return [self._weighted_token_choice(shape)]

    def update_weights(
        self, shape: str, tokens: List[str], score: float
    ) -> None:
        """Update token weights based on feedback score.

        Uses a simple linear update rule:
            reward = (score - 0.5) * 2   (maps [0, 1] to [-1, 1])
            learning_rate = 0.08
            weight += lr * reward

        Weights are clamped to [0.1, 10.0].

        Parameters
        ----------
        shape:
            Trajectory shape name.
        tokens:
            Tokens from the generated expression.
        score:
            Feedback score in [0, 1].
        """
        if shape not in self._token_weights:
            return
        lr = 0.08
        reward = (score - 0.5) * 2.0
        for token in tokens:
            if token in self._token_weights[shape]:
                new_w = self._token_weights[shape][token] + lr * reward
                self._token_weights[shape][token] = max(0.1, min(10.0, new_w))

    def get_weights(self, shape: str) -> Dict[str, float]:
        """Return current token weights for a shape (copy).

        Parameters
        ----------
        shape:
            Trajectory shape name.

        Returns
        -------
        Dict mapping token -> weight. Empty dict if shape unknown.
        """
        return dict(self._token_weights.get(shape, {}))
