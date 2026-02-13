"""Teacher model evaluation and Gate 1 quality check.

Evaluates teacher model predictions against coherence and validity
thresholds, providing a formal gate to proceed to student distillation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from eisv_lumen.eval.metrics import expression_trajectory_coherence
from eisv_lumen.training.teacher_train import OutputParseResult, validate_output

GATE1_COHERENCE_THRESHOLD = 0.933
GATE1_VALID_RATE_THRESHOLD = 0.90


@dataclass
class EvalResults:
    """Aggregated evaluation results from teacher model predictions."""

    mean_coherence: float
    valid_rate: float
    pattern_accuracy: float
    n_total: int
    n_valid: int
    per_shape_coherence: Dict[str, float] = field(default_factory=dict)
    diversity: float = 0.0


def evaluate_predictions(predictions: List[Dict]) -> EvalResults:
    """Evaluate a list of teacher model predictions.

    Parameters
    ----------
    predictions:
        List of dicts, each with keys:
        - ``shape`` (str): trajectory shape name
        - ``parsed`` (:class:`OutputParseResult`): parsed model output
        - ``expected_pattern`` (str, optional): ground-truth pattern for accuracy

    Returns
    -------
    An :class:`EvalResults` with aggregated metrics.
    """
    if not predictions:
        return EvalResults(
            mean_coherence=0.0,
            valid_rate=0.0,
            pattern_accuracy=0.0,
            n_total=0,
            n_valid=0,
            per_shape_coherence={},
            diversity=0.0,
        )

    n_total = len(predictions)
    n_valid = 0
    coherence_scores: List[float] = []
    per_shape_scores: Dict[str, List[float]] = defaultdict(list)
    pattern_matches = 0
    pattern_count = 0
    all_tokens: List[str] = []

    for pred in predictions:
        shape = pred["shape"]
        parsed: OutputParseResult = pred["parsed"]

        # Validity check
        if validate_output(parsed):
            n_valid += 1

        # Coherence: computed on eisv_tokens against shape
        if parsed.eisv_tokens:
            score = expression_trajectory_coherence(shape, parsed.eisv_tokens)
            coherence_scores.append(score)
            per_shape_scores[shape].append(score)
            all_tokens.extend(parsed.eisv_tokens)

        # Pattern accuracy
        expected = pred.get("expected_pattern")
        if expected is not None:
            pattern_count += 1
            if parsed.pattern == expected:
                pattern_matches += 1

    mean_coherence = (
        sum(coherence_scores) / len(coherence_scores)
        if coherence_scores
        else 0.0
    )

    valid_rate = n_valid / n_total if n_total > 0 else 0.0

    pattern_accuracy = (
        pattern_matches / pattern_count if pattern_count > 0 else 0.0
    )

    per_shape_coherence = {
        shape: sum(scores) / len(scores)
        for shape, scores in per_shape_scores.items()
    }

    diversity = (
        len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0
    )

    return EvalResults(
        mean_coherence=mean_coherence,
        valid_rate=valid_rate,
        pattern_accuracy=pattern_accuracy,
        n_total=n_total,
        n_valid=n_valid,
        per_shape_coherence=per_shape_coherence,
        diversity=diversity,
    )


def check_gate1(results: EvalResults) -> Tuple[bool, List[str]]:
    """Check whether evaluation results pass Gate 1.

    Gate 1 requires:
    - ``mean_coherence > GATE1_COHERENCE_THRESHOLD`` (0.933)
    - ``valid_rate >= GATE1_VALID_RATE_THRESHOLD`` (0.90)

    Parameters
    ----------
    results:
        Evaluation results from :func:`evaluate_predictions`.

    Returns
    -------
    Tuple of (passed, reasons) where ``passed`` is True if both criteria
    are met, and ``reasons`` is a list of failure reason strings (empty
    if passed).
    """
    reasons: List[str] = []

    if results.mean_coherence <= GATE1_COHERENCE_THRESHOLD:
        reasons.append(
            f"Mean coherence {results.mean_coherence:.4f} "
            f"<= threshold {GATE1_COHERENCE_THRESHOLD}"
        )

    if results.valid_rate < GATE1_VALID_RATE_THRESHOLD:
        reasons.append(
            f"Valid rate {results.valid_rate:.4f} "
            f"< threshold {GATE1_VALID_RATE_THRESHOLD}"
        )

    passed = len(reasons) == 0
    return passed, reasons
