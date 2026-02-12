"""Tests for trajectory shape classifier."""

from eisv_lumen.shapes.shape_classes import TrajectoryShape, classify_trajectory
from eisv_lumen.extract.derivatives import compute_trajectory_window


def _make_states(e_vals, i_vals=None, s_vals=None, v_vals=None, dt=60.0):
    """Helper: build state list from value sequences."""
    n = len(e_vals)
    i_vals = i_vals or [0.5] * n
    s_vals = s_vals or [0.3] * n
    v_vals = v_vals or [0.1] * n
    return [
        {"t": i * dt, "E": e_vals[i], "I": i_vals[i], "S": s_vals[i], "V": v_vals[i]}
        for i in range(n)
    ]


class TestTrajectoryShapeEnum:
    def test_has_nine_members(self):
        assert len(TrajectoryShape) == 9

    def test_all_values_are_strings(self):
        for shape in TrajectoryShape:
            assert isinstance(shape.value, str)


class TestSettledPresence:
    def test_high_energy_stable(self):
        """Steady high E with flat derivatives -> settled_presence."""
        states = _make_states([0.8, 0.8, 0.8, 0.8, 0.8])
        window = compute_trajectory_window(states)
        assert classify_trajectory(window) == TrajectoryShape.SETTLED_PRESENCE

    def test_default_fallback(self):
        """Ambiguous state defaults to settled_presence."""
        states = _make_states([0.5, 0.5, 0.5, 0.5, 0.5])
        window = compute_trajectory_window(states)
        assert classify_trajectory(window) == TrajectoryShape.SETTLED_PRESENCE


class TestRisingEntropy:
    def test_increasing_entropy(self):
        states = _make_states(
            [0.5] * 5,
            s_vals=[0.1, 0.3, 0.5, 0.7, 0.9],
            dt=1.0,
        )
        window = compute_trajectory_window(states)
        assert classify_trajectory(window) == TrajectoryShape.RISING_ENTROPY


class TestFallingEnergy:
    def test_decreasing_energy(self):
        # E range 0.19 < _BASIN_JUMP (0.2) avoids basin_transition_down.
        # dt=0.5 amplifies derivatives so mean(dE) ~ -0.095 < -0.05.
        states = _make_states([0.50, 0.46, 0.42, 0.38, 0.31], dt=0.5)
        window = compute_trajectory_window(states)
        assert classify_trajectory(window) == TrajectoryShape.FALLING_ENERGY


class TestBasinTransitionDown:
    def test_energy_drops(self):
        states = _make_states([0.9, 0.7, 0.5, 0.3, 0.1])
        window = compute_trajectory_window(states)
        assert classify_trajectory(window) == TrajectoryShape.BASIN_TRANSITION_DOWN


class TestBasinTransitionUp:
    def test_energy_rises(self):
        states = _make_states([0.1, 0.3, 0.5, 0.7, 0.9])
        window = compute_trajectory_window(states)
        assert classify_trajectory(window) == TrajectoryShape.BASIN_TRANSITION_UP


class TestEntropySpikeRecovery:
    def test_spike_pattern(self):
        states = _make_states(
            [0.5] * 5,
            s_vals=[0.2, 0.5, 0.9, 0.5, 0.2],
        )
        window = compute_trajectory_window(states)
        assert classify_trajectory(window) == TrajectoryShape.ENTROPY_SPIKE_RECOVERY


class TestDriftDissonance:
    def test_ethical_drift_present(self):
        """States with ethical_drift > 0.3 -> drift_dissonance."""
        states = _make_states([0.5] * 5)
        for s in states:
            s["ethical_drift"] = 0.0
        states[2]["ethical_drift"] = 0.5  # spike above 0.3
        window = compute_trajectory_window(states)
        assert classify_trajectory(window) == TrajectoryShape.DRIFT_DISSONANCE


class TestVoidRising:
    def test_increasing_void(self):
        states = _make_states(
            [0.5] * 5,
            v_vals=[0.1, 0.3, 0.5, 0.7, 0.9],
            dt=1.0,
        )
        window = compute_trajectory_window(states)
        assert classify_trajectory(window) == TrajectoryShape.VOID_RISING


class TestConvergence:
    def test_approaching_attractor(self):
        """Decaying oscillation approaches fixed point."""
        states = _make_states(
            e_vals=[0.5, 0.52, 0.5, 0.51, 0.5],
            i_vals=[0.5, 0.5, 0.5, 0.5, 0.5],
            s_vals=[0.3, 0.3, 0.3, 0.3, 0.3],
            v_vals=[0.1, 0.1, 0.1, 0.1, 0.1],
        )
        window = compute_trajectory_window(states)
        result = classify_trajectory(window)
        # With tiny oscillations, derivatives and second derivatives are small
        assert result in (TrajectoryShape.CONVERGENCE, TrajectoryShape.SETTLED_PRESENCE)
