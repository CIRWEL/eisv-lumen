"""Publish EISV trajectory dataset to HuggingFace.

Extracts real Lumen data from anima.db, generates synthetic trajectories
to fill underrepresented shape classes, combines into a unified dataset,
and publishes to HuggingFace Hub.

Usage:
    python3 -m eisv_lumen.scripts.publish_dataset [--db-path PATH] [--repo-id REPO] [--dry-run] [--min-per-shape N]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

from eisv_lumen.extract.assembler import assemble_dataset
from eisv_lumen.extract.lumen_expressions import extract_primitive_history
from eisv_lumen.extract.lumen_states import extract_state_history
from eisv_lumen.shapes.shape_classes import TrajectoryShape

DEFAULT_DB_PATH = "/Users/cirwel/.anima/anima.db"
DEFAULT_REPO_ID = "hikewa/unitares-eisv-trajectories"

ALL_SHAPE_VALUES = [s.value for s in TrajectoryShape]


def build_combined_dataset(
    db_path: str,
    min_per_shape: int = 50,
    window_size: int = 20,
    stride: int = 10,
    seed: int = 42,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """Build combined real + synthetic dataset.

    Parameters
    ----------
    db_path:
        Path to the anima.db SQLite database.
    min_per_shape:
        Minimum number of records per shape class.  Shapes with fewer
        real records will be augmented with synthetic trajectories.
    window_size:
        Number of EISV state snapshots per trajectory window.
    stride:
        Step size between consecutive windows.
    seed:
        Random seed for synthetic generation reproducibility.

    Returns
    -------
    Tuple of (records, metadata) where metadata includes:
    - real_count, synthetic_count, total_count
    - shape_distribution (with provenance breakdown)
    - shapes_filled (list of shapes that needed synthetic augmentation)
    """
    # 1. Extract real data
    print("Extracting state_history...", file=sys.stderr)
    states = extract_state_history(db_path, compute_eisv=True)
    print(f"  -> {len(states)} states", file=sys.stderr)

    print("Extracting primitive_history...", file=sys.stderr)
    expressions = extract_primitive_history(db_path)
    print(f"  -> {len(expressions)} expressions", file=sys.stderr)

    print(f"Assembling dataset (window={window_size}, stride={stride})...", file=sys.stderr)
    real_records = assemble_dataset(states, expressions, window_size, stride)
    print(f"  -> {len(real_records)} real trajectory windows", file=sys.stderr)

    # 2. Generate synthetic data for underrepresented shapes
    print("Filling missing/underrepresented shapes with synthetic data...", file=sys.stderr)
    from eisv_lumen.synthetic.trajectory_generator import fill_missing_shapes

    synthetic_records = fill_missing_shapes(
        real_records, min_per_shape=min_per_shape, seed=seed,
    )
    print(f"  -> {len(synthetic_records)} synthetic records generated", file=sys.stderr)

    # 3. Combine
    all_records = real_records + synthetic_records

    # 4. Build metadata
    real_shape_counts = Counter(r["shape"] for r in real_records)
    synthetic_shape_counts = Counter(r["shape"] for r in synthetic_records)
    total_shape_counts = Counter(r["shape"] for r in all_records)

    shapes_filled = sorted(synthetic_shape_counts.keys())

    shape_distribution: Dict[str, Dict[str, int]] = {}
    for shape in sorted(ALL_SHAPE_VALUES):
        real_n = real_shape_counts.get(shape, 0)
        synth_n = synthetic_shape_counts.get(shape, 0)
        total_n = total_shape_counts.get(shape, 0)
        if total_n > 0:
            shape_distribution[shape] = {
                "real": real_n,
                "synthetic": synth_n,
                "total": total_n,
            }

    metadata: Dict[str, Any] = {
        "real_count": len(real_records),
        "synthetic_count": len(synthetic_records),
        "total_count": len(all_records),
        "window_size": window_size,
        "stride": stride,
        "min_per_shape": min_per_shape,
        "shape_distribution": shape_distribution,
        "shapes_filled": shapes_filled,
    }

    return all_records, metadata


def publish_to_hf(
    records: List[Dict],
    metadata: Dict[str, Any],
    repo_id: str = DEFAULT_REPO_ID,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Publish dataset to HuggingFace Hub.

    If dry_run=True, validates and returns what would be published
    without importing datasets/huggingface_hub.

    Otherwise, pushes to HF Hub.

    Returns publish result dict.
    """
    from eisv_lumen.publish.hf_dataset import (
        trajectories_to_hf_format,
        generate_dataset_card,
    )

    # Convert to HF format
    hf_data = trajectories_to_hf_format(records)
    shape_counts = dict(Counter(hf_data["shape"]))

    # Generate dataset card
    card = generate_dataset_card(
        dataset_name=repo_id,
        n_records=len(records),
        shape_counts=shape_counts,
    )

    if dry_run:
        return {
            "status": "dry_run",
            "repo_id": repo_id,
            "n_records": len(records),
            "shape_counts": shape_counts,
            "card_length": len(card),
            "columns": list(hf_data.keys()),
            "metadata": metadata,
        }

    # Actually publish -- only import heavy libraries when needed
    from datasets import Dataset
    from huggingface_hub import HfApi

    dataset = Dataset.from_dict(hf_data)
    dataset.push_to_hub(repo_id, private=False)

    # Upload dataset card
    api = HfApi()
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    return {
        "status": "published",
        "repo_id": repo_id,
        "n_records": len(records),
        "shape_counts": shape_counts,
        "url": f"https://huggingface.co/datasets/{repo_id}",
    }


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Publish EISV trajectory dataset to HuggingFace",
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"Path to anima.db (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace repo ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without publishing",
    )
    parser.add_argument(
        "--min-per-shape",
        type=int,
        default=50,
        help="Minimum records per shape class (default: 50)",
    )
    args = parser.parse_args()

    records, metadata = build_combined_dataset(
        args.db_path, min_per_shape=args.min_per_shape,
    )
    print(json.dumps(metadata, indent=2))

    result = publish_to_hf(
        records, metadata, repo_id=args.repo_id, dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
