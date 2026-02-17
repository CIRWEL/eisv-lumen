"""Synthetic trajectory generator for all 9 EISV shape classes.

Generates deterministic EISV trajectories that classify correctly through
the rule-based classifier in ``eisv_lumen.shapes.shape_classes``.

Each shape-specific generator produces a list of ``{t, E, I, S, V}`` dicts
whose derivatives match the classification thresholds.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

from eisv_lumen.extract.derivatives import compute_trajectory_window
from eisv_lumen.shapes.shape_classes import TrajectoryShape, classify_trajectory

# The classifier threshold for derivative significance.
_THRESH = 0.05
# Basin-jump threshold (E range).
_BASIN_JUMP = 0.2


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))


def _add_noise(
    states: List[Dict[str, float]],
    rng: random.Random,
    scale: float = 0.005,
) -> List[Dict[str, float]]:
    """Add small Gaussian noise to EISV values, preserving t and clamping to [0, 1]."""
    result = []
    for s in states:
        noisy = {"t": s["t"]}
        for dim in ("E", "I", "S", "V"):
            noisy[dim] = _clamp(s[dim] + rng.gauss(0.0, scale))
        # Preserve ethical_drift if present
        if "ethical_drift" in s:
            noisy["ethical_drift"] = s["ethical_drift"]
        result.append(noisy)
    return result


def _linspace(start: float, end: float, n: int) -> List[float]:
    """Return n evenly spaced values from start to end (inclusive)."""
    if n == 1:
        return [start]
    return [start + (end - start) * i / (n - 1) for i in range(n)]


def _max_dt_for_slope(n: int, max_value_range: float, target_rate: float) -> float:
    """Compute the maximum dt such that a linear ramp of *max_value_range*
    over *n* points produces a per-second derivative of at least *target_rate*.

    derivative = value_range / ((n-1) * dt) >= target_rate
    => dt <= value_range / (target_rate * (n-1))
    """
    return max_value_range / (target_rate * (n - 1))


# ---------------------------------------------------------------------------
# Shape-specific generators
# ---------------------------------------------------------------------------
# Each returns a tuple of (states, noise_scale) where noise_scale is the
# recommended noise for this shape (0.0 for shapes that need exact values).
# ---------------------------------------------------------------------------


def _gen_settled_presence(
    n: int, dt: float, rng: random.Random,
) -> tuple[List[Dict[str, float]], float]:
    """Flat trajectory with zero dynamics.

    Uses no noise so that has_dynamics remains False, preventing
    accidental classification as convergence.
    """
    e_base = 0.65 + rng.uniform(0.0, 0.15)
    i_base = 0.60 + rng.uniform(0.0, 0.15)
    s_base = 0.15 + rng.uniform(0.0, 0.10)
    v_base = 0.05 + rng.uniform(0.0, 0.05)
    states = [
        {"t": i * dt, "E": e_base, "I": i_base, "S": s_base, "V": v_base}
        for i in range(n)
    ]
    return states, 0.0  # zero noise for truly settled


def _gen_rising_entropy(
    n: int, dt: float, rng: random.Random,
) -> tuple[List[Dict[str, float]], float]:
    """Entropy (S) increases steadily; need mean dS > 0.05.

    Must avoid higher-priority triggers: E range < 0.2, S monotonic
    (no interior max so entropy_spike_recovery is not triggered).
    """
    # Effective dt: ensure the ramp is feasible within [0, 1]
    # Total S change = (n-1) * eff_dt * target_deriv
    # Need total_change <= 0.9 (leave margin within [0,1])
    eff_dt = min(dt, _max_dt_for_slope(n, 0.85, _THRESH + 0.01))
    target_deriv = _THRESH + 0.01  # 0.06 per second
    total_s_change = target_deriv * (n - 1) * eff_dt

    s_start = 0.05 + rng.uniform(0.0, 0.10)
    s_end = _clamp(s_start + total_s_change, 0.0, 0.95)
    # Re-derive actual total to handle clamping
    total_s_change = s_end - s_start

    s_vals = _linspace(s_start, s_end, n)

    e_base = 0.50 + rng.uniform(0.0, 0.05)
    i_base = 0.55 + rng.uniform(0.0, 0.10)
    v_base = 0.08 + rng.uniform(0.0, 0.02)

    states = [
        {"t": i * eff_dt, "E": e_base, "I": i_base, "S": s_vals[i], "V": v_base}
        for i in range(n)
    ]
    return states, 0.001


def _gen_falling_energy(
    n: int, dt: float, rng: random.Random,
) -> tuple[List[Dict[str, float]], float]:
    """Energy decreases: mean dE < -0.05, but E range < 0.2.

    The feasibility constraint is:
        threshold * (n-1) * dt < 0.2
    If the caller's dt violates this, we use a smaller effective dt.
    """
    # Maximum dt for which falling_energy is feasible
    max_feasible = _max_dt_for_slope(n, 0.19, _THRESH + 0.005)
    eff_dt = min(dt, max_feasible)

    target_deriv = _THRESH + 0.01
    total_e_drop = target_deriv * (n - 1) * eff_dt
    # Ensure range stays strictly under _BASIN_JUMP
    total_e_drop = min(total_e_drop, 0.19)

    e_start = 0.55 + rng.uniform(0.0, 0.10)
    e_end = e_start - total_e_drop

    e_vals = _linspace(e_start, e_end, n)

    s_base = 0.20 + rng.uniform(0.0, 0.05)
    i_base = 0.55 + rng.uniform(0.0, 0.10)
    v_base = 0.08 + rng.uniform(0.0, 0.02)

    states = [
        {"t": i * eff_dt, "E": e_vals[i], "I": i_base, "S": s_base, "V": v_base}
        for i in range(n)
    ]
    return states, 0.001


def _gen_basin_transition_down(
    n: int, dt: float, rng: random.Random,
) -> tuple[List[Dict[str, float]], float]:
    """E drops substantially: range >= 0.2, mean dE < 0, first E > last E."""
    e_start = 0.80 + rng.uniform(0.0, 0.10)
    e_end = e_start - 0.25 - rng.uniform(0.0, 0.10)
    e_end = _clamp(e_end, 0.05, 0.55)

    e_vals = _linspace(e_start, e_end, n)

    s_base = 0.20 + rng.uniform(0.0, 0.05)
    i_base = 0.55 + rng.uniform(0.0, 0.10)
    v_base = 0.08 + rng.uniform(0.0, 0.02)

    states = [
        {"t": i * dt, "E": e_vals[i], "I": i_base, "S": s_base, "V": v_base}
        for i in range(n)
    ]
    return states, 0.002


def _gen_basin_transition_up(
    n: int, dt: float, rng: random.Random,
) -> tuple[List[Dict[str, float]], float]:
    """E rises substantially: range >= 0.2, mean dE > 0, last E > first E."""
    e_start = 0.30 + rng.uniform(0.0, 0.10)
    e_end = e_start + 0.25 + rng.uniform(0.0, 0.10)
    e_end = _clamp(e_end, 0.60, 0.95)

    e_vals = _linspace(e_start, e_end, n)

    s_base = 0.20 + rng.uniform(0.0, 0.05)
    i_base = 0.55 + rng.uniform(0.0, 0.10)
    v_base = 0.08 + rng.uniform(0.0, 0.02)

    states = [
        {"t": i * dt, "E": e_vals[i], "I": i_base, "S": s_base, "V": v_base}
        for i in range(n)
    ]
    return states, 0.002


def _gen_entropy_spike_recovery(
    n: int, dt: float, rng: random.Random,
) -> tuple[List[Dict[str, float]], float]:
    """S spikes mid-window then recovers.

    Classifier rule: s_range >= 0.2 AND max S is at an interior index.
    Must keep E range < 0.2 to avoid basin_transition taking priority.
    """
    s_base = 0.15 + rng.uniform(0.0, 0.05)
    s_peak = s_base + 0.30 + rng.uniform(0.0, 0.15)
    s_peak = _clamp(s_peak, 0.0, 0.95)

    # Place peak in the middle third of the window
    peak_idx = n // 2 + rng.randint(-max(1, n // 6), max(1, n // 6))
    peak_idx = max(1, min(n - 2, peak_idx))  # ensure interior

    s_vals = []
    for i in range(n):
        if i <= peak_idx:
            frac = i / peak_idx if peak_idx > 0 else 0.0
            s_vals.append(s_base + (s_peak - s_base) * frac)
        else:
            remaining = n - 1 - peak_idx
            frac = (i - peak_idx) / remaining if remaining > 0 else 1.0
            s_vals.append(s_peak - (s_peak - s_base) * frac)

    e_base = 0.55 + rng.uniform(0.0, 0.05)
    i_base = 0.55 + rng.uniform(0.0, 0.10)
    v_base = 0.08 + rng.uniform(0.0, 0.02)

    states = [
        {"t": i * dt, "E": e_base, "I": i_base, "S": s_vals[i], "V": v_base}
        for i in range(n)
    ]
    return states, 0.002


def _gen_drift_dissonance(
    n: int, dt: float, rng: random.Random,
) -> tuple[List[Dict[str, float]], float]:
    """Ethical drift exceeds 0.3 at some point.

    Classifier rule: max(ethical_drift) > 0.3.
    Must keep E range < 0.2 and S range < 0.2.
    """
    e_base = 0.50 + rng.uniform(0.0, 0.05)
    i_base = 0.35 + rng.uniform(0.0, 0.10)
    s_base = 0.20 + rng.uniform(0.0, 0.05)
    v_base = 0.08 + rng.uniform(0.0, 0.02)

    spike_idx = n // 2 + rng.randint(-1, 1)
    spike_idx = max(1, min(n - 2, spike_idx))

    states = []
    for i in range(n):
        drift = 0.05 + rng.uniform(0.0, 0.05)
        if i == spike_idx:
            drift = 0.35 + rng.uniform(0.0, 0.20)
        states.append({
            "t": i * dt,
            "E": e_base,
            "I": i_base,
            "S": s_base,
            "V": v_base,
            "ethical_drift": drift,
        })

    return states, 0.0  # no noise to avoid disturbing E/S ranges


def _gen_void_rising(
    n: int, dt: float, rng: random.Random,
) -> tuple[List[Dict[str, float]], float]:
    """V increases: mean dV > 0.05.

    Must keep E range < 0.2, S range < 0.2, and mean dS <= 0.05.
    Void_rising is rule 5, rising_entropy is rule 6 — void_rising
    has priority when both could match.
    """
    # Feasible if total V change < 0.95 (V starts near 0, ends near 1)
    eff_dt = min(dt, _max_dt_for_slope(n, 0.85, _THRESH + 0.01))
    target_deriv = _THRESH + 0.01
    total_v_change = target_deriv * (n - 1) * eff_dt

    v_start = 0.05 + rng.uniform(0.0, 0.05)
    v_end = _clamp(v_start + total_v_change, 0.0, 0.95)

    v_vals = _linspace(v_start, v_end, n)

    e_base = 0.55 + rng.uniform(0.0, 0.05)
    i_base = 0.55 + rng.uniform(0.0, 0.10)
    s_base = 0.20 + rng.uniform(0.0, 0.05)

    states = [
        {"t": i * eff_dt, "E": e_base, "I": i_base, "S": s_base, "V": v_vals[i]}
        for i in range(n)
    ]
    return states, 0.001


def _gen_convergence(
    n: int, dt: float, rng: random.Random,
) -> tuple[List[Dict[str, float]], float]:
    """All derivatives small but non-zero, with decaying oscillation.

    Classifier rule: all |derivatives| < 0.05, all |second_derivatives| < 0.05,
    but has_dynamics (some |derivative| > 1e-9).

    Strategy: tiny decaying oscillation around attractors.
    """
    e_attr = 0.60 + rng.uniform(0.0, 0.05)
    i_attr = 0.55 + rng.uniform(0.0, 0.05)
    s_attr = 0.25 + rng.uniform(0.0, 0.05)
    v_attr = 0.10 + rng.uniform(0.0, 0.03)

    period = n
    # Max derivative from oscillation: amplitude * 2*pi / (period * dt)
    # Want this < 0.04 (margin under 0.05)
    max_amplitude = 0.04 * period * dt / (2 * math.pi)
    amplitude = min(max_amplitude, 0.08) * 0.5

    # Ensure enough dynamics for has_dynamics check (derivative > 1e-9)
    min_amplitude = max(1e-7, 1e-8 * dt)
    amplitude = max(amplitude, min_amplitude)

    decay_rate = 2.0 / n

    states = []
    for i in range(n):
        t = i * dt
        decay = math.exp(-decay_rate * i)
        phase = 2 * math.pi * i / period
        osc_e = amplitude * decay * math.sin(phase)
        osc_i = amplitude * decay * math.cos(phase)
        osc_s = amplitude * 0.5 * decay * math.sin(phase + 1.0)
        osc_v = amplitude * 0.3 * decay * math.cos(phase + 2.0)

        states.append({
            "t": t,
            "E": _clamp(e_attr + osc_e),
            "I": _clamp(i_attr + osc_i),
            "S": _clamp(s_attr + osc_s),
            "V": _clamp(v_attr + osc_v),
        })

    return states, 0.0  # no extra noise -- oscillation provides dynamics


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_SHAPE_GENERATORS = {
    TrajectoryShape.SETTLED_PRESENCE.value: _gen_settled_presence,
    TrajectoryShape.RISING_ENTROPY.value: _gen_rising_entropy,
    TrajectoryShape.FALLING_ENERGY.value: _gen_falling_energy,
    TrajectoryShape.BASIN_TRANSITION_DOWN.value: _gen_basin_transition_down,
    TrajectoryShape.BASIN_TRANSITION_UP.value: _gen_basin_transition_up,
    TrajectoryShape.ENTROPY_SPIKE_RECOVERY.value: _gen_entropy_spike_recovery,
    TrajectoryShape.DRIFT_DISSONANCE.value: _gen_drift_dissonance,
    TrajectoryShape.VOID_RISING.value: _gen_void_rising,
    TrajectoryShape.CONVERGENCE.value: _gen_convergence,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_trajectory(
    shape: str,
    n_points: int = 20,
    dt: float = 1.0,
    seed: Optional[int] = None,
    *,
    noise_scale: Optional[float] = None,
    _max_retries: int = 20,
) -> List[Dict[str, float]]:
    """Generate a synthetic EISV trajectory that classifies as *shape*.

    Parameters
    ----------
    shape:
        One of the 9 ``TrajectoryShape`` value strings.
    n_points:
        Number of state snapshots in the trajectory.
    dt:
        Time step between consecutive snapshots (seconds).  Some shapes
        may use a smaller effective dt internally when the requested dt
        makes the shape mathematically infeasible (e.g. ``falling_energy``
        requires ``dt < 0.2 / (0.05 * (n-1))``).
    seed:
        Random seed for reproducibility.  ``None`` for non-deterministic.
    noise_scale:
        Standard deviation of Gaussian noise added to EISV values.
        ``None`` uses the shape-specific default.
    _max_retries:
        Internal: maximum regeneration attempts before raising.

    Returns
    -------
    List of ``{t, E, I, S, V}`` dicts (and ``ethical_drift`` for
    drift_dissonance).  All EISV values are clamped to [0, 1].

    Raises
    ------
    ValueError
        If *shape* is not a recognised shape name.
    RuntimeError
        If a trajectory passing classification cannot be produced within
        the retry budget.
    """
    if shape not in _SHAPE_GENERATORS:
        raise ValueError(
            f"Unknown shape {shape!r}.  "
            f"Must be one of: {sorted(_SHAPE_GENERATORS)}"
        )

    generator = _SHAPE_GENERATORS[shape]
    classified = None

    for attempt in range(_max_retries):
        if seed is not None:
            attempt_seed = seed + attempt * 7919
        else:
            attempt_seed = None

        rng = random.Random(attempt_seed)
        raw_states, shape_noise = generator(n_points, dt, rng)

        effective_noise = noise_scale if noise_scale is not None else shape_noise

        if effective_noise > 0:
            noise_rng = random.Random(
                (attempt_seed + 1) if attempt_seed is not None else None,
            )
            states = _add_noise(raw_states, noise_rng, scale=effective_noise)
        else:
            states = raw_states

        window = compute_trajectory_window(states)
        classified = classify_trajectory(window)
        if classified.value == shape:
            return states

    raise RuntimeError(
        f"Failed to generate trajectory for shape {shape!r} after "
        f"{_max_retries} attempts (last classified as {classified.value!r})."
    )


def generate_dataset(
    shape_counts: Dict[str, int],
    n_points: int = 20,
    dt: float = 1.0,
    base_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate multiple synthetic trajectory records.

    Parameters
    ----------
    shape_counts:
        Mapping from shape name to desired count, e.g.
        ``{"settled_presence": 10, "rising_entropy": 5}``.
    n_points:
        Points per trajectory.
    dt:
        Time step (seconds).
    base_seed:
        Seed for the first trajectory; subsequent trajectories use
        incrementing seeds for reproducibility.

    Returns
    -------
    List of dataset records matching the format produced by
    ``eisv_lumen.extract.assembler.build_trajectory_records``:
    ``{shape, states, derivatives, second_derivatives, t_start, t_end, provenance}``.
    """
    records: List[Dict[str, Any]] = []
    offset = 0

    for shape_name, count in sorted(shape_counts.items()):
        for i in range(count):
            seed = base_seed + offset
            states = generate_trajectory(
                shape_name, n_points=n_points, dt=dt, seed=seed,
            )
            window = compute_trajectory_window(states)
            classified = classify_trajectory(window)

            records.append({
                "shape": classified.value,
                "states": window["states"],
                "derivatives": window["derivatives"],
                "second_derivatives": window["second_derivatives"],
                "t_start": states[0]["t"],
                "t_end": states[-1]["t"],
                "provenance": "synthetic",
            })
            offset += 1

    return records


def fill_missing_shapes(
    real_records: List[Dict],
    min_per_shape: int = 50,
    n_points: int = 20,
    dt: float = 1.0,
    seed: int = 42,
) -> List[Dict]:
    """Generate synthetic records for underrepresented shapes.

    Analyses *real_records* to count occurrences of each shape, then
    generates synthetic trajectories to bring every shape up to at least
    *min_per_shape*.

    Parameters
    ----------
    real_records:
        Existing dataset records (each must have a ``"shape"`` key).
    min_per_shape:
        Minimum number of records desired per shape.
    n_points:
        Points per synthetic trajectory.
    dt:
        Time step (seconds).
    seed:
        Base random seed.

    Returns
    -------
    List of **only** the newly generated synthetic records.  The caller
    is responsible for combining these with the original *real_records*.
    """
    shape_counts: Dict[str, int] = {}
    for shape in TrajectoryShape:
        shape_counts[shape.value] = 0
    for rec in real_records:
        name = rec.get("shape", "")
        if name in shape_counts:
            shape_counts[name] += 1

    needed: Dict[str, int] = {}
    for shape_name, existing in shape_counts.items():
        deficit = min_per_shape - existing
        if deficit > 0:
            needed[shape_name] = deficit

    if not needed:
        return []

    return generate_dataset(
        needed, n_points=n_points, dt=dt, base_seed=seed,
    )
