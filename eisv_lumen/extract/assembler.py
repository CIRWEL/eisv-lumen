"""Dataset assembly pipeline combining extracts into HuggingFace-compatible records.

Combines EISV state histories and primitive expression histories into
classified trajectory windows suitable for model training and benchmarking.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from eisv_lumen.extract.derivatives import compute_trajectory_window
from eisv_lumen.shapes.shape_classes import classify_trajectory, TrajectoryShape


def _parse_timestamp(ts: Any) -> float:
    """Convert a timestamp to float epoch seconds.

    Supports float/int (returned as-is) and ISO-8601 strings.
    """
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        dt = datetime.fromisoformat(ts)
        # If naive (no tzinfo), assume UTC for consistent epoch conversion
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    raise TypeError(f"Unsupported timestamp type: {type(ts)}")


def states_to_eisv_series(states: List[Dict]) -> List[Dict[str, float]]:
    """Convert extracted state_history records (with eisv key) to derivative-ready format.

    Each input dict must have 'timestamp' (ISO string or epoch float) and 'eisv'
    dict with E, I, S, V keys.

    Returns list of ``{t: float_epoch, E, I, S, V}`` sorted by ``t``.
    """
    series: List[Dict[str, float]] = []
    for record in states:
        eisv = record["eisv"]
        series.append({
            "t": _parse_timestamp(record["timestamp"]),
            "E": float(eisv["E"]),
            "I": float(eisv["I"]),
            "S": float(eisv["S"]),
            "V": float(eisv["V"]),
        })
    series.sort(key=lambda s: s["t"])
    return series


def build_trajectory_records(
    eisv_series: List[Dict[str, float]],
    window_size: int = 10,
    stride: int = 5,
) -> List[Dict[str, Any]]:
    """Slide a window across EISV series, classify each window.

    Returns list of dicts with keys:
        shape, states, derivatives, second_derivatives, t_start, t_end, provenance
    """
    records: List[Dict[str, Any]] = []
    n = len(eisv_series)

    for start in range(0, n - window_size + 1, stride):
        window_states = eisv_series[start : start + window_size]
        trajectory = compute_trajectory_window(window_states)
        shape = classify_trajectory(trajectory)

        records.append({
            "shape": shape.value,
            "states": trajectory["states"],
            "derivatives": trajectory["derivatives"],
            "second_derivatives": trajectory["second_derivatives"],
            "t_start": window_states[0]["t"],
            "t_end": window_states[-1]["t"],
            "provenance": "lumen_real",
        })

    return records


def align_expressions_to_trajectories(
    trajectory_records: List[Dict],
    expressions: List[Dict],
) -> List[Dict]:
    """Match primitive expressions to trajectory windows by timestamp overlap.

    For each trajectory record, finds expressions whose timestamp falls within
    ``[t_start, t_end]``. Adds an ``"expressions"`` key to each trajectory
    record containing the list of matching expression dicts.

    Returns the enriched trajectory records.
    """
    # Pre-parse expression timestamps for efficient matching
    parsed_exprs: List[tuple[float, Dict]] = []
    for expr in expressions:
        t = _parse_timestamp(expr["timestamp"])
        parsed_exprs.append((t, expr))
    # Sort by timestamp for potential optimisation
    parsed_exprs.sort(key=lambda x: x[0])

    for rec in trajectory_records:
        t_start = rec["t_start"]
        t_end = rec["t_end"]
        matched = [
            expr for t, expr in parsed_exprs if t_start <= t <= t_end
        ]
        rec["expressions"] = matched

    return trajectory_records


def assemble_dataset(
    state_records: List[Dict],
    expression_records: Optional[List[Dict]] = None,
    window_size: int = 10,
    stride: int = 5,
) -> List[Dict[str, Any]]:
    """Full pipeline: states -> EISV series -> trajectory windows -> aligned expressions.

    Parameters
    ----------
    state_records:
        Output of ``extract_state_history(compute_eisv=True)``.
    expression_records:
        Output of ``extract_primitive_history()`` (optional).
    window_size:
        Number of state snapshots per trajectory window.
    stride:
        Step size between consecutive windows.

    Returns list of fully assembled dataset records.
    """
    eisv_series = states_to_eisv_series(state_records)
    trajectory_records = build_trajectory_records(eisv_series, window_size, stride)

    if expression_records:
        align_expressions_to_trajectories(trajectory_records, expression_records)
    else:
        for rec in trajectory_records:
            rec["expressions"] = []

    return trajectory_records
