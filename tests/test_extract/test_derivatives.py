"""Tests for EISV derivative computation with irregular timesteps."""

import pytest
from eisv_lumen.extract.derivatives import (
    compute_derivatives,
    compute_second_derivatives,
    compute_trajectory_window,
)


class TestComputeDerivativesBasic:
    def test_compute_derivatives_basic(self):
        states = [
            {"t": 0.0, "E": 1.0, "I": 2.0, "S": 3.0, "V": 4.0},
            {"t": 300.0, "E": 4.0, "I": 5.0, "S": 0.0, "V": 10.0},
        ]
        derivs = compute_derivatives(states)
        assert len(derivs) == 1
        dt = 300.0
        assert derivs[0]["t"] == pytest.approx(300.0)
        assert derivs[0]["dE"] == pytest.approx((4.0 - 1.0) / dt)
        assert derivs[0]["dI"] == pytest.approx((5.0 - 2.0) / dt)
        assert derivs[0]["dS"] == pytest.approx((0.0 - 3.0) / dt)
        assert derivs[0]["dV"] == pytest.approx((10.0 - 4.0) / dt)


class TestComputeDerivativesIrregularTimesteps:
    def test_compute_derivatives_irregular_timesteps(self):
        states = [
            {"t": 0.0, "E": 0.0, "I": 0.0, "S": 0.0, "V": 0.0},
            {"t": 60.0, "E": 6.0, "I": 12.0, "S": 18.0, "V": 24.0},
            {"t": 600.0, "E": 60.0, "I": 120.0, "S": 180.0, "V": 240.0},
        ]
        derivs = compute_derivatives(states)
        assert len(derivs) == 2
        assert derivs[0]["t"] == pytest.approx(60.0)
        assert derivs[0]["dE"] == pytest.approx(6.0 / 60.0)
        assert derivs[0]["dI"] == pytest.approx(12.0 / 60.0)
        assert derivs[0]["dS"] == pytest.approx(18.0 / 60.0)
        assert derivs[0]["dV"] == pytest.approx(24.0 / 60.0)
        assert derivs[1]["t"] == pytest.approx(600.0)
        assert derivs[1]["dE"] == pytest.approx((60.0 - 6.0) / 540.0)
        assert derivs[1]["dI"] == pytest.approx((120.0 - 12.0) / 540.0)
        assert derivs[1]["dS"] == pytest.approx((180.0 - 18.0) / 540.0)
        assert derivs[1]["dV"] == pytest.approx((240.0 - 24.0) / 540.0)


class TestComputeDerivativesHandlesZeroDt:
    def test_compute_derivatives_handles_zero_dt(self):
        states = [
            {"t": 0.0, "E": 1.0, "I": 2.0, "S": 3.0, "V": 4.0},
            {"t": 0.0, "E": 1.5, "I": 2.5, "S": 3.5, "V": 4.5},
            {"t": 100.0, "E": 5.0, "I": 6.0, "S": 7.0, "V": 8.0},
        ]
        derivs = compute_derivatives(states)
        assert len(derivs) == 1
        assert derivs[0]["t"] == pytest.approx(100.0)


class TestComputeSecondDerivatives:
    def test_compute_second_derivatives(self):
        first_derivs = [
            {"t": 100.0, "dE": 0.1, "dI": 0.2, "dS": 0.3, "dV": 0.4},
            {"t": 200.0, "dE": 0.3, "dI": 0.0, "dS": 0.1, "dV": 0.8},
        ]
        second_derivs = compute_second_derivatives(first_derivs)
        assert len(second_derivs) == 1
        dt = 100.0
        assert second_derivs[0]["t"] == pytest.approx(200.0)
        assert second_derivs[0]["d2E"] == pytest.approx((0.3 - 0.1) / dt)
        assert second_derivs[0]["d2I"] == pytest.approx((0.0 - 0.2) / dt)
        assert second_derivs[0]["d2S"] == pytest.approx((0.1 - 0.3) / dt)
        assert second_derivs[0]["d2V"] == pytest.approx((0.8 - 0.4) / dt)


class TestComputeTrajectoryWindow:
    def test_compute_trajectory_window(self):
        states = [
            {"t": 0.0, "E": 0.0, "I": 0.0, "S": 0.0, "V": 0.0},
            {"t": 10.0, "E": 1.0, "I": 2.0, "S": 3.0, "V": 4.0},
            {"t": 20.0, "E": 3.0, "I": 5.0, "S": 7.0, "V": 9.0},
            {"t": 30.0, "E": 6.0, "I": 9.0, "S": 12.0, "V": 15.0},
        ]
        window = compute_trajectory_window(states)
        assert "states" in window
        assert "derivatives" in window
        assert "second_derivatives" in window
        assert len(window["states"]) == 4
        assert len(window["derivatives"]) == 3
        assert len(window["second_derivatives"]) == 2
        assert window["states"] == states
        for d in window["derivatives"]:
            assert "t" in d
            assert "dE" in d
            assert "dI" in d
            assert "dS" in d
            assert "dV" in d
        for d2 in window["second_derivatives"]:
            assert "t" in d2
            assert "d2E" in d2
            assert "d2I" in d2
            assert "d2S" in d2
            assert "d2V" in d2
