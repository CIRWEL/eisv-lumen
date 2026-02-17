"""Tests for eisv_lumen.training.dataset_builder — dataset construction and splitting."""

from __future__ import annotations

from collections import Counter

import pytest

from eisv_lumen.shapes.shape_classes import TrajectoryShape
from eisv_lumen.training.data_prep import TrainingExample
from eisv_lumen.training.dataset_builder import (
    build_training_dataset,
    split_dataset,
)


def _make_real_records(shapes: list[str], n_per: int = 5) -> list[dict]:
    """Create minimal real records with the required keys."""
    from eisv_lumen.synthetic.trajectory_generator import generate_trajectory
    from eisv_lumen.extract.derivatives import compute_trajectory_window

    records = []
    seed = 100
    for shape in shapes:
        for i in range(n_per):
            states = generate_trajectory(shape, seed=seed + i)
            window = compute_trajectory_window(states)
            records.append({
                "shape": shape,
                "states": window["states"],
                "derivatives": window["derivatives"],
                "second_derivatives": window["second_derivatives"],
                "t_start": states[0]["t"],
                "t_end": states[-1]["t"],
                "provenance": "real",
            })
        seed += 1000
    return records


# ---------------------------------------------------------------------------
# TestBuildTrainingDataset
# ---------------------------------------------------------------------------


class TestBuildTrainingDataset:
    def test_returns_list_of_examples(self):
        """build_training_dataset should return a list of TrainingExample."""
        examples = build_training_dataset([], min_per_shape=3, seed=42)
        assert isinstance(examples, list)
        assert all(isinstance(ex, TrainingExample) for ex in examples)

    def test_all_shapes_represented(self):
        """Every shape class should appear in the dataset."""
        examples = build_training_dataset([], min_per_shape=3, seed=42)
        shapes_present = {ex.shape for ex in examples}
        expected = {s.value for s in TrajectoryShape}
        assert shapes_present == expected

    def test_min_per_shape_enforced(self):
        """Each shape should have at least min_per_shape examples."""
        min_count = 5
        examples = build_training_dataset([], min_per_shape=min_count, seed=42)
        shape_counts = Counter(ex.shape for ex in examples)
        for shape in TrajectoryShape:
            assert shape_counts[shape.value] >= min_count, (
                f"{shape.value} has only {shape_counts[shape.value]} examples"
            )

    def test_real_records_included(self):
        """Real records should be included in the output."""
        real = _make_real_records(["settled_presence"], n_per=3)
        examples = build_training_dataset(real, min_per_shape=2, seed=42)
        settled_count = sum(1 for ex in examples if ex.shape == "settled_presence")
        # Should have at least the 3 real ones
        assert settled_count >= 3


# ---------------------------------------------------------------------------
# TestSplitDataset
# ---------------------------------------------------------------------------


class TestSplitDataset:
    def _get_examples(self, n: int = 100) -> list[TrainingExample]:
        """Get a set of examples for split testing."""
        return build_training_dataset([], min_per_shape=max(3, n // 9), seed=42)

    def test_split_ratios(self):
        """Train/val/test split should roughly match the requested ratios."""
        examples = self._get_examples(100)
        train, val, test = split_dataset(examples, train_ratio=0.8, val_ratio=0.1, seed=42)
        total = len(examples)
        # Allow some tolerance due to rounding and stratification
        assert abs(len(train) / total - 0.8) < 0.15
        assert abs(len(val) / total - 0.1) < 0.15
        assert abs(len(test) / total - 0.1) < 0.15

    def test_no_overlap(self):
        """Train, val, and test sets should not share examples."""
        examples = self._get_examples(100)
        train, val, test = split_dataset(examples, seed=42)
        # Use input_text as a unique identifier since TrainingExample is not hashable
        train_ids = {ex.input_text + ex.output_text for ex in train}
        val_ids = {ex.input_text + ex.output_text for ex in val}
        test_ids = {ex.input_text + ex.output_text for ex in test}
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0
        assert len(train) + len(val) + len(test) == len(examples)

    def test_deterministic(self):
        """Same seed should produce the same split."""
        examples = self._get_examples(50)
        train1, val1, test1 = split_dataset(examples, seed=99)
        train2, val2, test2 = split_dataset(examples, seed=99)
        assert [ex.shape for ex in train1] == [ex.shape for ex in train2]
        assert [ex.shape for ex in val1] == [ex.shape for ex in val2]
        assert [ex.shape for ex in test1] == [ex.shape for ex in test2]

    def test_stratified_by_shape(self):
        """Each split should contain examples from multiple shapes."""
        examples = self._get_examples(100)
        train, val, test = split_dataset(examples, seed=42)
        # Train should have most shapes
        train_shapes = {ex.shape for ex in train}
        assert len(train_shapes) >= 7  # at least most of the 9 shapes
        # Val and test should have some shape diversity too
        val_shapes = {ex.shape for ex in val}
        test_shapes = {ex.shape for ex in test}
        assert len(val_shapes) >= 3
        assert len(test_shapes) >= 3
