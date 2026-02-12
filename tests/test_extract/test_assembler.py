"""Tests for dataset assembly pipeline."""

import pytest
from eisv_lumen.extract.assembler import (
    states_to_eisv_series,
    build_trajectory_records,
    align_expressions_to_trajectories,
    assemble_dataset,
)
from eisv_lumen.shapes.shape_classes import TrajectoryShape


def _fake_state_records(n=20, dt=60.0):
    """Create fake state_history records with EISV."""
    records = []
    for i in range(n):
        t = 1000.0 + i * dt
        records.append({
            "timestamp": t,  # use float epoch directly
            "warmth": 0.7,
            "clarity": 0.6,
            "stability": 0.8,
            "presence": 0.9,
            "eisv": {"E": 0.7, "I": 0.6, "S": 0.2, "V": 0.03},
        })
    return records


def _fake_expressions(n=5, t_start=1000.0, dt=120.0):
    """Create fake primitive_history records."""
    return [
        {
            "id": i + 1,
            "timestamp": t_start + i * dt,
            "tokens": ["~warmth~"],
            "category_pattern": ["STATE"],
            "state_at_generation": {"warmth": 0.7, "brightness": 0.6, "stability": 0.8, "presence": 0.9},
            "score": 0.6,
            "feedback_signals": {},
        }
        for i in range(n)
    ]


class TestStatesToEisvSeries:
    def test_returns_correct_keys(self):
        records = _fake_state_records(5)
        series = states_to_eisv_series(records)
        assert len(series) == 5
        for s in series:
            assert set(s.keys()) == {"t", "E", "I", "S", "V"}

    def test_sorted_by_time(self):
        records = _fake_state_records(5)
        # Shuffle
        records = records[::-1]
        series = states_to_eisv_series(records)
        times = [s["t"] for s in series]
        assert times == sorted(times)

    def test_iso_string_timestamp(self):
        """Support ISO-8601 strings."""
        records = [{
            "timestamp": "2026-02-12T12:00:00",
            "eisv": {"E": 0.5, "I": 0.5, "S": 0.3, "V": 0.1},
        }]
        series = states_to_eisv_series(records)
        assert len(series) == 1
        assert isinstance(series[0]["t"], float)


class TestBuildTrajectoryRecords:
    def test_window_count(self):
        series = states_to_eisv_series(_fake_state_records(20))
        records = build_trajectory_records(series, window_size=10, stride=5)
        # 20 items, window=10, stride=5: windows at [0:10], [5:15], [10:20] = 3
        assert len(records) == 3

    def test_record_structure(self):
        series = states_to_eisv_series(_fake_state_records(20))
        records = build_trajectory_records(series, window_size=10, stride=5)
        rec = records[0]
        assert "shape" in rec
        assert "states" in rec
        assert "derivatives" in rec
        assert "second_derivatives" in rec
        assert "t_start" in rec
        assert "t_end" in rec
        assert rec["provenance"] == "lumen_real"

    def test_shape_is_valid(self):
        series = states_to_eisv_series(_fake_state_records(20))
        records = build_trajectory_records(series, window_size=10, stride=5)
        for rec in records:
            assert rec["shape"] in [s.value for s in TrajectoryShape]


class TestAlignExpressions:
    def test_matches_by_timestamp(self):
        series = states_to_eisv_series(_fake_state_records(20))
        traj_recs = build_trajectory_records(series, window_size=10, stride=5)
        exprs = _fake_expressions(5, t_start=1000.0, dt=120.0)
        enriched = align_expressions_to_trajectories(traj_recs, exprs)
        # At least some trajectories should have expressions
        total_matched = sum(len(r["expressions"]) for r in enriched)
        assert total_matched > 0

    def test_no_expressions(self):
        series = states_to_eisv_series(_fake_state_records(20))
        traj_recs = build_trajectory_records(series, window_size=10, stride=5)
        enriched = align_expressions_to_trajectories(traj_recs, [])
        for r in enriched:
            assert r["expressions"] == []


class TestAssembleDataset:
    def test_full_pipeline(self):
        states = _fake_state_records(20)
        exprs = _fake_expressions(5)
        records = assemble_dataset(states, exprs, window_size=10, stride=5)
        assert len(records) > 0
        for rec in records:
            assert "shape" in rec
            assert "expressions" in rec

    def test_without_expressions(self):
        states = _fake_state_records(20)
        records = assemble_dataset(states, window_size=10, stride=5)
        assert len(records) > 0
        for rec in records:
            assert rec["expressions"] == []
