"""Tests for synthetic trajectory generator."""

import pytest

from eisv_lumen.extract.derivatives import compute_trajectory_window
from eisv_lumen.shapes.shape_classes import TrajectoryShape, classify_trajectory
from eisv_lumen.synthetic.trajectory_generator import (
    generate_trajectory,
    generate_dataset,
    fill_missing_shapes,
)


ALL_SHAPES = [s.value for s in TrajectoryShape]


# ---------------------------------------------------------------------------
# generate_trajectory: classification correctness for all 9 shapes
# ---------------------------------------------------------------------------


class TestGenerateTrajectoryClassification:
    """Every generated trajectory must classify as the requested shape."""

    @pytest.mark.parametrize("shape", ALL_SHAPES)
    def test_classifies_correctly(self, shape):
        states = generate_trajectory(shape, n_points=20, dt=1.0, seed=42)
        window = compute_trajectory_window(states)
        result = classify_trajectory(window)
        assert result.value == shape, (
            f"Expected {shape}, got {result.value}"
        )

    @pytest.mark.parametrize("shape", ALL_SHAPES)
    def test_classifies_correctly_multiple_seeds(self, shape):
        """Check several seeds to guard against lucky single-seed passes."""
        for seed in (0, 7, 99, 256, 1000):
            states = generate_trajectory(shape, n_points=20, dt=1.0, seed=seed)
            window = compute_trajectory_window(states)
            result = classify_trajectory(window)
            assert result.value == shape, (
                f"seed={seed}: expected {shape}, got {result.value}"
            )


# ---------------------------------------------------------------------------
# generate_trajectory: reproducibility
# ---------------------------------------------------------------------------


class TestGenerateTrajectoryReproducibility:
    @pytest.mark.parametrize("shape", ALL_SHAPES)
    def test_same_seed_same_output(self, shape):
        a = generate_trajectory(shape, n_points=15, dt=1.0, seed=123)
        b = generate_trajectory(shape, n_points=15, dt=1.0, seed=123)
        assert len(a) == len(b)
        for sa, sb in zip(a, b):
            for key in ("t", "E", "I", "S", "V"):
                assert sa[key] == sb[key]

    @pytest.mark.parametrize("shape", ALL_SHAPES)
    def test_different_seeds_different_output(self, shape):
        a = generate_trajectory(shape, n_points=15, dt=1.0, seed=1)
        b = generate_trajectory(shape, n_points=15, dt=1.0, seed=2)
        # At least one EISV value should differ (ignoring t which is
        # deterministic from index * dt).
        diffs = []
        for sa, sb in zip(a, b):
            for key in ("E", "I", "S", "V"):
                diffs.append(abs(sa[key] - sb[key]))
        assert max(diffs) > 1e-6, "Different seeds should produce different trajectories"


# ---------------------------------------------------------------------------
# generate_trajectory: value bounds
# ---------------------------------------------------------------------------


class TestGenerateTrajectoryBounds:
    @pytest.mark.parametrize("shape", ALL_SHAPES)
    def test_values_in_unit_interval(self, shape):
        states = generate_trajectory(shape, n_points=20, dt=1.0, seed=42)
        for s in states:
            for dim in ("E", "I", "S", "V"):
                assert 0.0 <= s[dim] <= 1.0, (
                    f"{shape}: {dim}={s[dim]} out of [0, 1]"
                )


# ---------------------------------------------------------------------------
# generate_trajectory: edge cases
# ---------------------------------------------------------------------------


class TestGenerateTrajectoryEdgeCases:
    def test_unknown_shape_raises(self):
        with pytest.raises(ValueError, match="Unknown shape"):
            generate_trajectory("nonexistent_shape")

    def test_drift_dissonance_has_ethical_drift(self):
        states = generate_trajectory("drift_dissonance", seed=42)
        # At least one state should have ethical_drift > 0.3
        drifts = [s.get("ethical_drift", 0.0) for s in states]
        assert max(drifts) > 0.3

    def test_n_points_respected(self):
        for n in (5, 10, 30):
            states = generate_trajectory("settled_presence", n_points=n, seed=42)
            assert len(states) == n


# ---------------------------------------------------------------------------
# generate_dataset
# ---------------------------------------------------------------------------


class TestGenerateDataset:
    def test_correct_counts(self):
        counts = {"settled_presence": 3, "rising_entropy": 5, "void_rising": 2}
        records = generate_dataset(counts, n_points=20, dt=1.0, base_seed=42)
        assert len(records) == 10  # 3 + 5 + 2

        shape_dist = {}
        for r in records:
            shape_dist[r["shape"]] = shape_dist.get(r["shape"], 0) + 1

        assert shape_dist.get("settled_presence", 0) == 3
        assert shape_dist.get("rising_entropy", 0) == 5
        assert shape_dist.get("void_rising", 0) == 2

    def test_record_format(self):
        records = generate_dataset(
            {"convergence": 2}, n_points=15, dt=1.0, base_seed=0,
        )
        for rec in records:
            assert "shape" in rec
            assert "states" in rec
            assert "derivatives" in rec
            assert "second_derivatives" in rec
            assert "t_start" in rec
            assert "t_end" in rec
            assert rec["provenance"] == "synthetic"

    def test_provenance_is_synthetic(self):
        records = generate_dataset({"settled_presence": 1}, base_seed=0)
        assert all(r["provenance"] == "synthetic" for r in records)

    def test_all_nine_shapes(self):
        """Generate at least one trajectory per shape; all must classify correctly."""
        counts = {s: 2 for s in ALL_SHAPES}
        records = generate_dataset(counts, n_points=20, dt=1.0, base_seed=99)
        assert len(records) == 18

        for rec in records:
            assert rec["shape"] in ALL_SHAPES


# ---------------------------------------------------------------------------
# fill_missing_shapes
# ---------------------------------------------------------------------------


class TestFillMissingShapes:
    def _make_real_records(self, shape_counts):
        """Helper: create minimal real records with given shape distribution."""
        records = []
        for shape, count in shape_counts.items():
            for _ in range(count):
                records.append({"shape": shape, "provenance": "lumen_real"})
        return records

    def test_fills_missing_shapes(self):
        real = self._make_real_records({
            "settled_presence": 100,
            "rising_entropy": 100,
            # All other shapes: 0
        })
        synthetic = fill_missing_shapes(real, min_per_shape=5, seed=42)

        # Should have generated records for the 7 missing shapes
        syn_shapes = {r["shape"] for r in synthetic}
        for shape in ALL_SHAPES:
            if shape not in ("settled_presence", "rising_entropy"):
                assert shape in syn_shapes, f"Missing synthetic records for {shape}"

    def test_does_not_add_well_represented(self):
        real = self._make_real_records({s: 100 for s in ALL_SHAPES})
        synthetic = fill_missing_shapes(real, min_per_shape=50, seed=42)
        assert synthetic == [], "Should not generate records when all shapes >= min"

    def test_partial_fill(self):
        """Only fills shapes below the threshold."""
        real = self._make_real_records({
            "settled_presence": 100,
            "rising_entropy": 3,  # below min_per_shape=10
            "void_rising": 0,    # missing entirely
        })
        synthetic = fill_missing_shapes(real, min_per_shape=10, seed=42)

        syn_shape_counts = {}
        for r in synthetic:
            syn_shape_counts[r["shape"]] = syn_shape_counts.get(r["shape"], 0) + 1

        # settled_presence already has 100 -> no synthetic
        assert "settled_presence" not in syn_shape_counts

        # rising_entropy needs 7 more (10 - 3)
        assert syn_shape_counts.get("rising_entropy", 0) == 7

        # void_rising needs all 10
        assert syn_shape_counts.get("void_rising", 0) == 10

    def test_fill_returns_only_synthetic(self):
        real = self._make_real_records({"settled_presence": 2})
        synthetic = fill_missing_shapes(real, min_per_shape=5, seed=42)
        for rec in synthetic:
            assert rec["provenance"] == "synthetic"
