"""Dataset builder for teacher model training.

Constructs balanced training datasets by combining real trajectory records
with synthetically generated ones, ensuring all 9 shape classes are
adequately represented.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from eisv_lumen.shapes.shape_classes import TrajectoryShape
from eisv_lumen.extract.derivatives import compute_trajectory_window
from eisv_lumen.synthetic.trajectory_generator import generate_trajectory
from eisv_lumen.training.data_prep import TrainingExample, build_training_example


@dataclass
class DatasetStats:
    """Summary statistics for a built training dataset."""

    total: int
    per_shape: Dict[str, int]
    real_count: int
    synthetic_count: int


def build_training_dataset(
    real_records: List[Dict[str, Any]],
    min_per_shape: int = 50,
    seed: int = 42,
) -> List[TrainingExample]:
    """Build a balanced training dataset from real and synthetic trajectories.

    Converts real trajectory records to training examples, then generates
    synthetic trajectories for any shape class that is underrepresented
    (below *min_per_shape*).

    Parameters
    ----------
    real_records:
        List of trajectory record dicts with at least ``shape``, ``states``,
        ``derivatives``, and ``second_derivatives`` keys.
    min_per_shape:
        Minimum number of training examples per shape class.
    seed:
        Random seed for reproducible synthetic generation.

    Returns
    -------
    List of :class:`TrainingExample` instances covering all 9 shape classes.
    """
    examples: List[TrainingExample] = []
    shape_counts: Dict[str, int] = {s.value: 0 for s in TrajectoryShape}
    example_seed = seed

    # Process real records
    for rec in real_records:
        shape = rec["shape"]
        window = {
            "states": rec["states"],
            "derivatives": rec["derivatives"],
            "second_derivatives": rec["second_derivatives"],
        }
        example = build_training_example(shape, window, seed=example_seed)
        examples.append(example)
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
        example_seed += 1

    # Fill underrepresented shapes with synthetic trajectories
    synth_seed_offset = seed + 10000
    for shape in TrajectoryShape:
        deficit = min_per_shape - shape_counts[shape.value]
        for i in range(max(0, deficit)):
            traj_seed = synth_seed_offset + i
            states = generate_trajectory(shape.value, seed=traj_seed)
            window = compute_trajectory_window(states)
            example = build_training_example(
                shape.value, window, seed=traj_seed + 50000
            )
            examples.append(example)
            shape_counts[shape.value] += 1
        synth_seed_offset += 10000

    return examples


def split_dataset(
    examples: List[TrainingExample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[TrainingExample], List[TrainingExample], List[TrainingExample]]:
    """Split a dataset into train, validation, and test sets.

    Uses stratified splitting to preserve the shape distribution across
    all three sets.

    Parameters
    ----------
    examples:
        Full list of training examples.
    train_ratio:
        Fraction of data for training (default 0.8).
    val_ratio:
        Fraction of data for validation (default 0.1).
        Test ratio is ``1 - train_ratio - val_ratio``.
    seed:
        Random seed for deterministic splitting.

    Returns
    -------
    Tuple of (train, validation, test) lists.
    """
    rng = random.Random(seed)

    # Group by shape for stratified splitting
    by_shape: Dict[str, List[TrainingExample]] = defaultdict(list)
    for ex in examples:
        by_shape[ex.shape].append(ex)

    train: List[TrainingExample] = []
    val: List[TrainingExample] = []
    test: List[TrainingExample] = []

    for shape in sorted(by_shape.keys()):
        group = by_shape[shape]
        rng.shuffle(group)

        n = len(group)
        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio)) if n > 2 else 0
        # Adjust if we overallocated
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1) if n > 1 else 0
        n_test = n - n_train - n_val

        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    return train, val, test
