"""Tests for the incremental dataset growth script."""

from __future__ import annotations

from collections import Counter
from unittest.mock import MagicMock, patch

import pytest

from eisv_lumen.scripts.grow_dataset import (
    dedup_records,
    extract_new_windows,
    merge_and_fill,
)
from eisv_lumen.shapes.shape_classes import TrajectoryShape
from eisv_lumen.synthetic.trajectory_generator import generate_dataset

ALL_SHAPES = [s.value for s in TrajectoryShape]


def _make_records(shape_counts: dict[str, int], provenance: str = "lumen_real") -> list[dict]:
    """Build trajectory records using the synthetic generator."""
    records = generate_dataset(shape_counts, n_points=20, dt=1.0, base_seed=200)
    for rec in records:
        rec["provenance"] = provenance
        rec["expressions"] = []
    return records


def _make_minimal_records(
    shape: str, count: int, t_start_base: float = 0.0, stride: float = 10.0,
    provenance: str = "lumen_real",
) -> list[dict]:
    """Build minimal records with incrementing t_start values."""
    records = []
    for i in range(count):
        records.append({
            "shape": shape,
            "states": [{"t": float(j), "E": 0.5, "I": 0.5, "S": 0.2, "V": 0.1} for j in range(5)],
            "derivatives": [{"dE": 0.0, "dI": 0.0, "dS": 0.0, "dV": 0.0}],
            "second_derivatives": [],
            "t_start": t_start_base + i * stride,
            "t_end": t_start_base + i * stride + 19.0,
            "provenance": provenance,
            "expressions": [],
        })
    return records


# ---------------------------------------------------------------------------
# Tests: dedup_records
# ---------------------------------------------------------------------------

class TestDedupRecords:

    def test_no_overlap(self):
        existing = _make_minimal_records("convergence", 3, t_start_base=0.0)
        new = _make_minimal_records("convergence", 3, t_start_base=100.0)
        result = dedup_records(existing, new)
        assert len(result) == 3

    def test_full_overlap(self):
        existing = _make_minimal_records("convergence", 3, t_start_base=0.0)
        new = _make_minimal_records("convergence", 3, t_start_base=0.0)
        result = dedup_records(existing, new)
        assert len(result) == 0

    def test_partial_overlap(self):
        existing = _make_minimal_records("convergence", 3, t_start_base=0.0, stride=10.0)
        # new records at t_start = 0, 10, 50 — first two overlap, third doesn't
        new = [
            _make_minimal_records("convergence", 1, t_start_base=0.0)[0],
            _make_minimal_records("convergence", 1, t_start_base=10.0)[0],
            _make_minimal_records("convergence", 1, t_start_base=50.0)[0],
        ]
        result = dedup_records(existing, new)
        assert len(result) == 1
        assert result[0]["t_start"] == 50.0

    def test_empty_existing(self):
        new = _make_minimal_records("convergence", 3)
        result = dedup_records([], new)
        assert len(result) == 3

    def test_empty_new(self):
        existing = _make_minimal_records("convergence", 3)
        result = dedup_records(existing, [])
        assert len(result) == 0

    def test_tolerance_default(self):
        """Records within 0.5 seconds are considered duplicates."""
        existing = _make_minimal_records("convergence", 1, t_start_base=10.0)
        new = _make_minimal_records("convergence", 1, t_start_base=10.3)
        result = dedup_records(existing, new)
        assert len(result) == 0  # within tolerance

    def test_tolerance_boundary(self):
        """Records at exactly tolerance distance are not duplicates."""
        existing = _make_minimal_records("convergence", 1, t_start_base=10.0)
        new = _make_minimal_records("convergence", 1, t_start_base=10.6)
        result = dedup_records(existing, new)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Tests: extract_new_windows
# ---------------------------------------------------------------------------

class TestExtractNewWindows:

    @patch("eisv_lumen.scripts.grow_dataset.extract_state_history")
    @patch("eisv_lumen.scripts.grow_dataset.states_to_eisv_series")
    @patch("eisv_lumen.scripts.grow_dataset.build_trajectory_records")
    def test_filters_by_cutoff(self, mock_build, mock_series, mock_extract):
        """Only windows with t_start > after_t are returned."""
        mock_extract.return_value = [{"timestamp": i * 10.0, "eisv": {"E": 0.5, "I": 0.5, "S": 0.2, "V": 0.1}} for i in range(100)]
        mock_series.return_value = [{"t": i * 10.0, "E": 0.5, "I": 0.5, "S": 0.2, "V": 0.1} for i in range(100)]

        # Simulate windows at various t_start values
        mock_build.return_value = [
            {"t_start": 50.0, "t_end": 69.0, "shape": "convergence", "provenance": "lumen_real"},
            {"t_start": 100.0, "t_end": 119.0, "shape": "convergence", "provenance": "lumen_real"},
            {"t_start": 200.0, "t_end": 219.0, "shape": "convergence", "provenance": "lumen_real"},
        ]

        result = extract_new_windows("/fake/db", after_t=80.0)
        assert len(result) == 2
        assert all(r["t_start"] > 80.0 for r in result)

    @patch("eisv_lumen.scripts.grow_dataset.extract_state_history")
    @patch("eisv_lumen.scripts.grow_dataset.states_to_eisv_series")
    def test_returns_empty_when_no_new_data(self, mock_series, mock_extract):
        """Returns empty list when all data is before the cutoff."""
        mock_extract.return_value = [{"timestamp": i, "eisv": {"E": 0.5, "I": 0.5, "S": 0.2, "V": 0.1}} for i in range(5)]
        mock_series.return_value = [{"t": i, "E": 0.5, "I": 0.5, "S": 0.2, "V": 0.1} for i in range(5)]

        result = extract_new_windows("/fake/db", after_t=99999.0)
        assert result == []


# ---------------------------------------------------------------------------
# Tests: merge_and_fill
# ---------------------------------------------------------------------------

class TestMergeAndFill:

    def test_merges_real_records(self):
        """New real records are added to existing real records."""
        existing = _make_minimal_records("convergence", 10, provenance="lumen_real")
        existing += _make_minimal_records("convergence", 5, provenance="synthetic")
        new = _make_minimal_records("rising_entropy", 3, t_start_base=500.0)

        all_records, metadata = merge_and_fill(existing, new, min_per_shape=5)

        # Should have 10 real convergence + 3 new rising_entropy + synthetic fill
        assert metadata["new_windows_added"] == 3
        assert metadata["real_count"] == 13  # 10 existing real + 3 new

    def test_existing_synthetic_excluded_from_real_count(self):
        """Existing synthetic records are dropped; only real + new are kept."""
        existing = _make_minimal_records("convergence", 5, provenance="lumen_real")
        existing += _make_minimal_records("convergence", 100, provenance="synthetic")

        all_records, metadata = merge_and_fill(existing, [], min_per_shape=5)

        # Only the 5 real convergence records survive
        real_in_output = [r for r in all_records if r["provenance"] != "synthetic"]
        assert len(real_in_output) == 5

    def test_synthetic_fill_reaches_min_per_shape(self):
        """Synthetic fill brings all shapes to min_per_shape."""
        existing = _make_records({"convergence": 10}, provenance="lumen_real")
        all_records, metadata = merge_and_fill(existing, [], min_per_shape=15)

        shape_counts = Counter(r["shape"] for r in all_records)
        for shape in ALL_SHAPES:
            assert shape_counts.get(shape, 0) >= 15

    def test_metadata_shape_distribution(self):
        """Metadata includes per-shape real/synthetic/total breakdown."""
        existing = _make_records({"convergence": 10}, provenance="lumen_real")
        _, metadata = merge_and_fill(existing, [], min_per_shape=15)

        assert "shape_distribution" in metadata
        dist = metadata["shape_distribution"]
        assert "convergence" in dist
        assert dist["convergence"]["real"] == 10
        assert dist["convergence"]["synthetic"] == 5
        assert dist["convergence"]["total"] == 15
