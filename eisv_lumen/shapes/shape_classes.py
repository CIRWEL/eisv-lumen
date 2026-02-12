"""Rule-based trajectory shape classifier for EISV dynamics."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

_HIGH_BASIN_E = 0.6
_DERIV_THRESHOLD = 0.05
_BASIN_JUMP = 0.2


class TrajectoryShape(str, Enum):
    SETTLED_PRESENCE = "settled_presence"
    RISING_ENTROPY = "rising_entropy"
    FALLING_ENERGY = "falling_energy"
    BASIN_TRANSITION_DOWN = "basin_transition_down"
    BASIN_TRANSITION_UP = "basin_transition_up"
    ENTROPY_SPIKE_RECOVERY = "entropy_spike_recovery"
    DRIFT_DISSONANCE = "drift_dissonance"
    VOID_RISING = "void_rising"
    CONVERGENCE = "convergence"


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def classify_trajectory(window: Dict[str, Any]) -> TrajectoryShape:
    """Classify a trajectory window into one of 9 dynamical shape classes.

    Rules are checked in priority order; first match wins.
    """
    states = window["states"]
    derivs = window["derivatives"]
    second = window["second_derivatives"]

    # Extract value sequences
    e_vals = [s["E"] for s in states]
    s_vals = [s["S"] for s in states]

    e_range = max(e_vals) - min(e_vals)
    s_range = max(s_vals) - min(s_vals)

    de_vals = [d["dE"] for d in derivs] if derivs else [0.0]
    ds_vals = [d["dS"] for d in derivs] if derivs else [0.0]
    dv_vals = [d["dV"] for d in derivs] if derivs else [0.0]

    mean_de = _mean(de_vals)
    mean_ds = _mean(ds_vals)
    mean_dv = _mean(dv_vals)

    # 1. Basin transition down
    if e_range >= _BASIN_JUMP and mean_de < 0 and e_vals[0] > e_vals[-1]:
        return TrajectoryShape.BASIN_TRANSITION_DOWN

    # 2. Basin transition up
    if e_range >= _BASIN_JUMP and mean_de > 0 and e_vals[-1] > e_vals[0]:
        return TrajectoryShape.BASIN_TRANSITION_UP

    # 3. Entropy spike recovery
    if s_range >= _BASIN_JUMP:
        max_s_idx = s_vals.index(max(s_vals))
        if 0 < max_s_idx < len(s_vals) - 1:
            return TrajectoryShape.ENTROPY_SPIKE_RECOVERY

    # 4. Drift dissonance
    drift_vals = [s.get("ethical_drift", 0.0) for s in states]
    if max(drift_vals) > 0.3:
        return TrajectoryShape.DRIFT_DISSONANCE

    # 5. Void rising
    if mean_dv > _DERIV_THRESHOLD:
        return TrajectoryShape.VOID_RISING

    # 6. Rising entropy
    if mean_ds > _DERIV_THRESHOLD:
        return TrajectoryShape.RISING_ENTROPY

    # 7. Falling energy
    if mean_de < -_DERIV_THRESHOLD:
        return TrajectoryShape.FALLING_ENERGY

    # 8. Convergence: all derivatives and second derivatives are small in
    #    magnitude, but at least *some* dynamics are present (otherwise the
    #    system is already settled, not converging).
    if derivs and second:
        all_derivs_small = all(
            abs(d[k]) < _DERIV_THRESHOLD
            for d in derivs
            for k in ("dE", "dI", "dS", "dV")
        )
        all_second_small = all(
            abs(d[k]) < _DERIV_THRESHOLD
            for d in second
            for k in ("d2E", "d2I", "d2S", "d2V")
        )
        # Require some non-trivial dynamics to distinguish convergence from
        # already-settled states. If all derivatives are essentially zero the
        # system is not *approaching* an attractor -- it is already there.
        has_dynamics = any(
            abs(d[k]) > 1e-9
            for d in derivs
            for k in ("dE", "dI", "dS", "dV")
        )
        if all_derivs_small and all_second_small and has_dynamics:
            return TrajectoryShape.CONVERGENCE

    # 9. Default: settled presence
    return TrajectoryShape.SETTLED_PRESENCE
