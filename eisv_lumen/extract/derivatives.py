"""Compute EISV derivatives from state time series with irregular timesteps."""

from typing import Any, Dict, List

EISV_DIMS = ["E", "I", "S", "V"]


def compute_derivatives(states: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Finite-difference first derivatives from EISV state snapshots.

    Input: list of dicts with keys 't', 'E', 'I', 'S', 'V'.
    Returns: list of dicts with keys 't', 'dE', 'dI', 'dS', 'dV'.

    Each output entry corresponds to the interval ending at that timestamp.
    Pairs with dt=0 are skipped to avoid division by zero.
    """
    results: List[Dict[str, float]] = []
    prev = states[0]
    for i in range(1, len(states)):
        curr = states[i]
        dt = curr["t"] - prev["t"]
        if dt == 0.0:
            # Skip zero-length intervals; update prev to latest value at this time
            prev = curr
            continue
        entry: Dict[str, float] = {"t": curr["t"]}
        for dim in EISV_DIMS:
            entry[f"d{dim}"] = (curr[dim] - prev[dim]) / dt
        results.append(entry)
        prev = curr
    return results


def compute_second_derivatives(
    derivatives: List[Dict[str, float]],
) -> List[Dict[str, float]]:
    """Second derivatives from a first-derivative series.

    Input: list of dicts with keys 't', 'dE', 'dI', 'dS', 'dV'.
    Returns: list of dicts with keys 't', 'd2E', 'd2I', 'd2S', 'd2V'.
    """
    results: List[Dict[str, float]] = []
    for i in range(1, len(derivatives)):
        prev = derivatives[i - 1]
        curr = derivatives[i]
        dt = curr["t"] - prev["t"]
        if dt == 0.0:
            continue
        entry: Dict[str, float] = {"t": curr["t"]}
        for dim in EISV_DIMS:
            entry[f"d2{dim}"] = (curr[f"d{dim}"] - prev[f"d{dim}"]) / dt
        results.append(entry)
    return results


def compute_trajectory_window(
    states: List[Dict[str, float]],
) -> Dict[str, Any]:
    """Build a complete trajectory window from state snapshots.

    Returns a dict with:
        'states': the original state list
        'derivatives': first derivatives (n-1 entries for n states)
        'second_derivatives': second derivatives (n-2 entries for n states)
    """
    derivatives = compute_derivatives(states)
    second_derivatives = compute_second_derivatives(derivatives)
    return {
        "states": states,
        "derivatives": derivatives,
        "second_derivatives": second_derivatives,
    }
