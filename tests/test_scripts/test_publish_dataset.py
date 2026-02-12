"""Tests for the dataset publication script.

Tests build_combined_dataset and publish_to_hf functions using
synthetic data (no real database or HuggingFace credentials needed).
"""

from __future__ import annotations

import sys
from collections import Counter
from unittest.mock import MagicMock, patch

import pytest

from eisv_lumen.scripts.publish_dataset import (
    DEFAULT_DB_PATH,
    DEFAULT_REPO_ID,
    build_combined_dataset,
    publish_to_hf,
)
from eisv_lumen.shapes.shape_classes import TrajectoryShape
from eisv_lumen.synthetic.trajectory_generator import generate_dataset


# ---------------------------------------------------------------------------
# Helpers: build synthetic trajectory records
# ---------------------------------------------------------------------------

ALL_SHAPES = [s.value for s in TrajectoryShape]


def _make_real_records(shape_counts: dict[str, int]) -> list[dict]:
    """Build realistic trajectory records using the synthetic generator.

    Uses the actual trajectory generator to produce records with valid
    states, derivatives, and shape classifications -- but marks them
    with provenance ``"lumen_real"`` to simulate real data.
    """
    records = generate_dataset(shape_counts, n_points=20, dt=1.0, base_seed=100)
    for rec in records:
        rec["provenance"] = "lumen_real"
        rec["expressions"] = []
    return records


def _make_minimal_records(shape_counts: dict[str, int]) -> list[dict]:
    """Build minimal records that only carry the keys needed for testing."""
    records = []
    for shape, count in shape_counts.items():
        for i in range(count):
            records.append({
                "shape": shape,
                "states": [{"t": float(j), "E": 0.5, "I": 0.5, "S": 0.2, "V": 0.1} for j in range(5)],
                "derivatives": [{"dE": 0.0, "dI": 0.0, "dS": 0.0, "dV": 0.0}],
                "second_derivatives": [],
                "t_start": 0.0,
                "t_end": 4.0,
                "provenance": "lumen_real",
                "expressions": [],
            })
    return records


# ---------------------------------------------------------------------------
# Tests: build_combined_dataset
# ---------------------------------------------------------------------------

class TestBuildCombinedDataset:
    """Tests for build_combined_dataset metadata and synthetic augmentation."""

    @patch("eisv_lumen.scripts.publish_dataset.extract_primitive_history")
    @patch("eisv_lumen.scripts.publish_dataset.extract_state_history")
    @patch("eisv_lumen.scripts.publish_dataset.assemble_dataset")
    def test_metadata_structure(
        self,
        mock_assemble,
        mock_extract_states,
        mock_extract_expressions,
    ):
        """Metadata dict has all required keys."""
        # Provide real records for a few shapes, leaving others empty
        real = _make_real_records({"convergence": 60, "settled_presence": 60})
        mock_extract_states.return_value = []
        mock_extract_expressions.return_value = []
        mock_assemble.return_value = real

        records, metadata = build_combined_dataset(
            "/fake/db.db", min_per_shape=10, seed=42,
        )

        assert "real_count" in metadata
        assert "synthetic_count" in metadata
        assert "total_count" in metadata
        assert "shape_distribution" in metadata
        assert "shapes_filled" in metadata
        assert "window_size" in metadata
        assert "stride" in metadata
        assert "min_per_shape" in metadata

        assert metadata["real_count"] == len(real)
        assert metadata["total_count"] == metadata["real_count"] + metadata["synthetic_count"]

    @patch("eisv_lumen.scripts.publish_dataset.extract_primitive_history")
    @patch("eisv_lumen.scripts.publish_dataset.extract_state_history")
    @patch("eisv_lumen.scripts.publish_dataset.assemble_dataset")
    def test_synthetic_fills_missing_shapes(
        self,
        mock_assemble,
        mock_extract_states,
        mock_extract_expressions,
    ):
        """Shapes below min_per_shape are filled with synthetic records."""
        # Only provide convergence -- all others should get synthetic data
        real = _make_real_records({"convergence": 60})
        mock_extract_states.return_value = []
        mock_extract_expressions.return_value = []
        mock_assemble.return_value = real

        min_per = 10
        records, metadata = build_combined_dataset(
            "/fake/db.db", min_per_shape=min_per, seed=42,
        )

        assert metadata["synthetic_count"] > 0
        assert len(metadata["shapes_filled"]) > 0

        # Every shape should have at least min_per_shape records
        shape_counts = Counter(r["shape"] for r in records)
        for shape in ALL_SHAPES:
            assert shape_counts.get(shape, 0) >= min_per, (
                f"Shape {shape} has {shape_counts.get(shape, 0)} records, "
                f"expected >= {min_per}"
            )

    @patch("eisv_lumen.scripts.publish_dataset.extract_primitive_history")
    @patch("eisv_lumen.scripts.publish_dataset.extract_state_history")
    @patch("eisv_lumen.scripts.publish_dataset.assemble_dataset")
    def test_no_synthetic_when_all_shapes_sufficient(
        self,
        mock_assemble,
        mock_extract_states,
        mock_extract_expressions,
    ):
        """No synthetic records generated when all shapes meet threshold."""
        shape_counts = {shape: 60 for shape in ALL_SHAPES}
        real = _make_real_records(shape_counts)
        mock_extract_states.return_value = []
        mock_extract_expressions.return_value = []
        mock_assemble.return_value = real

        records, metadata = build_combined_dataset(
            "/fake/db.db", min_per_shape=50, seed=42,
        )

        assert metadata["synthetic_count"] == 0
        assert metadata["shapes_filled"] == []
        assert metadata["total_count"] == metadata["real_count"]

    @patch("eisv_lumen.scripts.publish_dataset.extract_primitive_history")
    @patch("eisv_lumen.scripts.publish_dataset.extract_state_history")
    @patch("eisv_lumen.scripts.publish_dataset.assemble_dataset")
    def test_shape_distribution_has_provenance_breakdown(
        self,
        mock_assemble,
        mock_extract_states,
        mock_extract_expressions,
    ):
        """Shape distribution includes real/synthetic/total per shape."""
        real = _make_real_records({"convergence": 30})
        mock_extract_states.return_value = []
        mock_extract_expressions.return_value = []
        mock_assemble.return_value = real

        _, metadata = build_combined_dataset(
            "/fake/db.db", min_per_shape=50, seed=42,
        )

        dist = metadata["shape_distribution"]
        # convergence should show real=30, synthetic=20, total=50
        assert "convergence" in dist
        assert dist["convergence"]["real"] == 30
        assert dist["convergence"]["synthetic"] == 20
        assert dist["convergence"]["total"] == 50


# ---------------------------------------------------------------------------
# Tests: publish_to_hf
# ---------------------------------------------------------------------------

class TestPublishToHf:
    """Tests for the publish_to_hf function."""

    def test_dry_run_returns_correct_structure(self):
        """Dry run returns all expected keys without publishing."""
        records = _make_minimal_records({"convergence": 5, "settled_presence": 3})
        metadata = {
            "real_count": 8,
            "synthetic_count": 0,
            "total_count": 8,
        }

        result = publish_to_hf(records, metadata, dry_run=True)

        assert result["status"] == "dry_run"
        assert result["repo_id"] == DEFAULT_REPO_ID
        assert result["n_records"] == 8
        assert "shape_counts" in result
        assert result["shape_counts"]["convergence"] == 5
        assert result["shape_counts"]["settled_presence"] == 3
        assert "card_length" in result
        assert result["card_length"] > 0
        assert "columns" in result
        expected_columns = [
            "shape", "eisv_states", "derivatives",
            "t_start", "t_end", "provenance", "tokens", "n_expressions",
        ]
        assert result["columns"] == expected_columns
        assert "metadata" in result

    def test_dry_run_custom_repo_id(self):
        """Dry run uses the custom repo ID."""
        records = _make_minimal_records({"convergence": 2})
        metadata = {"real_count": 2, "synthetic_count": 0, "total_count": 2}

        result = publish_to_hf(
            records, metadata, repo_id="test/custom-repo", dry_run=True,
        )
        assert result["repo_id"] == "test/custom-repo"

    def test_dry_run_does_not_import_heavy_libraries(self):
        """Dry run should not import datasets or huggingface_hub."""
        records = _make_minimal_records({"convergence": 2})
        metadata = {"real_count": 2, "synthetic_count": 0, "total_count": 2}

        # Remove these modules if they happen to be loaded
        modules_to_check = ["datasets", "huggingface_hub"]
        saved = {}
        for mod in modules_to_check:
            if mod in sys.modules:
                saved[mod] = sys.modules.pop(mod)

        try:
            result = publish_to_hf(records, metadata, dry_run=True)
            assert result["status"] == "dry_run"

            # Verify that the heavy libraries were not loaded
            for mod in modules_to_check:
                assert mod not in sys.modules, (
                    f"Module {mod!r} should not be imported in dry_run mode"
                )
        finally:
            # Restore any modules we removed
            sys.modules.update(saved)

    def test_dry_run_with_empty_records(self):
        """Dry run handles empty record list gracefully."""
        metadata = {"real_count": 0, "synthetic_count": 0, "total_count": 0}
        result = publish_to_hf([], metadata, dry_run=True)
        assert result["status"] == "dry_run"
        assert result["n_records"] == 0
        assert result["shape_counts"] == {}


# ---------------------------------------------------------------------------
# Tests: argparse defaults
# ---------------------------------------------------------------------------

class TestArgparseDefaults:
    """Tests for CLI argument defaults."""

    def test_default_db_path(self):
        """Default db_path matches the constant."""
        assert DEFAULT_DB_PATH == "/Users/cirwel/.anima/anima.db"

    def test_default_repo_id(self):
        """Default repo_id matches the constant."""
        assert DEFAULT_REPO_ID == "hikewa/unitares-eisv-trajectories"

    def test_argparse_defaults_match(self):
        """Argument parser defaults match the module constants."""
        import argparse

        # Import main to exercise the argparse setup without running
        from eisv_lumen.scripts.publish_dataset import main

        # Build the parser the same way main() does
        parser = argparse.ArgumentParser()
        parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
        parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--min-per-shape", type=int, default=50)

        args = parser.parse_args([])  # empty args => defaults

        assert args.db_path == DEFAULT_DB_PATH
        assert args.repo_id == DEFAULT_REPO_ID
        assert args.dry_run is False
        assert args.min_per_shape == 50
