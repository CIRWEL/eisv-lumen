"""Incremental dataset growth for EISV trajectories.

Downloads the current published dataset from HuggingFace, extracts new
Lumen data from anima.db (windows after the latest t_end in published data),
generates synthetic fill for underrepresented shapes, merges, and publishes.

Usage::

    python3 -m eisv_lumen.scripts.grow_dataset [OPTIONS]
    python3 -m eisv_lumen.scripts.grow_dataset --dry-run
    python3 -m eisv_lumen.scripts.grow_dataset --governance-url postgresql://...
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eisv_lumen.extract.assembler import (
    assemble_dataset,
    build_trajectory_records,
    states_to_eisv_series,
)
from eisv_lumen.extract.lumen_states import extract_state_history
from eisv_lumen.publish.hf_dataset import trajectories_to_hf_format
from eisv_lumen.shapes.shape_classes import TrajectoryShape

DEFAULT_DB_PATH = os.environ.get(
    "ANIMA_DB", os.path.join(Path.home(), ".anima", "anima.db"),
)
DEFAULT_REPO_ID = "hikewa/unitares-eisv-trajectories"
DEFAULT_PG_URL = os.environ.get(
    "DB_POSTGRES_URL", "postgresql://postgres:postgres@localhost:5432/governance",
)
ALL_SHAPES = [s.value for s in TrajectoryShape]


def get_published_cutoff(repo_id: str) -> float:
    """Load the published dataset and return the maximum t_end.

    Returns 0.0 if the dataset cannot be loaded.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(repo_id, split="train")
        max_t = max(ds["t_end"])
        return float(max_t)
    except Exception as exc:
        print(f"  Warning: could not load published dataset: {exc}", file=sys.stderr)
        return 0.0


def get_published_records(repo_id: str) -> List[Dict[str, Any]]:
    """Download the published dataset and convert back to record dicts.

    Returns a list of records with keys matching the assembler output format.
    """
    from datasets import load_dataset

    ds = load_dataset(repo_id, split="train")
    records: List[Dict[str, Any]] = []
    for row in ds:
        records.append({
            "shape": row["shape"],
            "states": json.loads(row["eisv_states"]),
            "derivatives": json.loads(row["derivatives"]),
            "second_derivatives": [],
            "t_start": row["t_start"],
            "t_end": row["t_end"],
            "provenance": row["provenance"],
            "expressions": [],
        })
    return records


def extract_new_windows(
    db_path: str,
    after_t: float,
    window_size: int = 20,
    stride: int = 10,
) -> List[Dict[str, Any]]:
    """Extract trajectory windows from anima.db that start after *after_t*.

    Only returns windows whose t_start > after_t, avoiding overlap with
    already-published data.
    """
    states = extract_state_history(db_path, compute_eisv=True)
    eisv_series = states_to_eisv_series(states)

    # Filter to states after the cutoff (with some overlap for windowing)
    overlap = window_size * 2  # keep some earlier states for window context
    cutoff_idx = 0
    for i, s in enumerate(eisv_series):
        if s["t"] > after_t:
            cutoff_idx = max(0, i - overlap)
            break
    else:
        return []  # no new data

    filtered_series = eisv_series[cutoff_idx:]
    if len(filtered_series) < window_size:
        return []

    all_windows = build_trajectory_records(filtered_series, window_size, stride)

    # Only keep windows that genuinely start after the cutoff
    new_windows = [w for w in all_windows if w["t_start"] > after_t]
    return new_windows


def extract_governance_windows(
    pg_url: str,
    agent_id: Optional[int] = None,
    since: Optional[float] = None,
    window_size: int = 20,
    stride: int = 10,
) -> List[Dict[str, Any]]:
    """Extract trajectory windows from the governance PostgreSQL database."""
    from eisv_lumen.extract.governance_states import extract_governance_history

    states = extract_governance_history(pg_url, agent_id=agent_id, since=since)
    if len(states) < window_size:
        print(f"  -> Only {len(states)} governance states (need {window_size})", file=sys.stderr)
        return []

    eisv_series = states_to_eisv_series(states)
    records = build_trajectory_records(eisv_series, window_size, stride)

    # Mark provenance as governance
    for r in records:
        r["provenance"] = "governance_real"

    return records


def dedup_records(
    existing: List[Dict[str, Any]],
    new: List[Dict[str, Any]],
    tolerance: float = 0.5,
) -> List[Dict[str, Any]]:
    """Remove records from *new* whose t_start overlaps with *existing*.

    Two records overlap if their t_start values are within *tolerance* seconds.
    """
    existing_starts = {r["t_start"] for r in existing}

    deduped = []
    for r in new:
        # Check if any existing record has a similar t_start
        is_dup = any(abs(r["t_start"] - es) < tolerance for es in existing_starts)
        if not is_dup:
            deduped.append(r)
    return deduped


def merge_and_fill(
    existing_records: List[Dict[str, Any]],
    new_records: List[Dict[str, Any]],
    min_per_shape: int = 2000,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Merge existing + new real records, then fill with synthetic.

    Returns (all_records, metadata).
    """
    from eisv_lumen.synthetic.trajectory_generator import fill_missing_shapes

    # Separate real and synthetic from existing
    existing_real = [r for r in existing_records if r["provenance"] != "synthetic"]

    # Combine real records (existing real + new)
    all_real = existing_real + new_records

    # Generate synthetic fill
    synthetic = fill_missing_shapes(all_real, min_per_shape=min_per_shape, seed=seed)

    all_records = all_real + synthetic

    # Build metadata
    real_counts = Counter(r["shape"] for r in all_real)
    synth_counts = Counter(r["shape"] for r in synthetic)
    total_counts = Counter(r["shape"] for r in all_records)

    shape_dist: Dict[str, Dict[str, int]] = {}
    for shape in sorted(ALL_SHAPES):
        total_n = total_counts.get(shape, 0)
        if total_n > 0:
            shape_dist[shape] = {
                "real": real_counts.get(shape, 0),
                "synthetic": synth_counts.get(shape, 0),
                "total": total_n,
            }

    metadata = {
        "real_count": len(all_real),
        "synthetic_count": len(synthetic),
        "total_count": len(all_records),
        "new_windows_added": len(new_records),
        "min_per_shape": min_per_shape,
        "shape_distribution": shape_dist,
    }

    return all_records, metadata


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Incrementally grow the EISV trajectory dataset",
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
        help="Show what would change without publishing",
    )
    parser.add_argument(
        "--min-per-shape",
        type=int,
        default=2000,
        help="Minimum records per shape class (default: 2000)",
    )
    parser.add_argument(
        "--governance-url",
        default=None,
        help="PostgreSQL URL for governance DB (omit to skip governance data)",
    )
    parser.add_argument(
        "--governance-agent-id",
        type=int,
        default=None,
        help="Only extract governance states for this identity_id",
    )
    args = parser.parse_args()

    # 1. Load published dataset
    print("Loading published dataset...", file=sys.stderr)
    cutoff = get_published_cutoff(args.repo_id)
    print(f"  -> Published cutoff t_end: {cutoff}", file=sys.stderr)

    existing_records = get_published_records(args.repo_id)
    print(f"  -> {len(existing_records)} existing records", file=sys.stderr)

    # 2. Extract new Lumen windows
    new_records: List[Dict[str, Any]] = []
    if os.path.exists(args.db_path):
        print(f"Extracting new Lumen windows (after t={cutoff})...", file=sys.stderr)
        lumen_new = extract_new_windows(args.db_path, cutoff)
        lumen_deduped = dedup_records(existing_records, lumen_new)
        print(
            f"  -> {len(lumen_new)} raw, {len(lumen_deduped)} after dedup",
            file=sys.stderr,
        )
        new_records.extend(lumen_deduped)
    else:
        print(f"  Skipping Lumen data (no file at {args.db_path})", file=sys.stderr)

    # 3. Extract governance windows (optional)
    if args.governance_url:
        print("Extracting governance windows...", file=sys.stderr)
        gov_records = extract_governance_windows(
            args.governance_url,
            agent_id=args.governance_agent_id,
            since=cutoff if cutoff > 0 else None,
        )
        gov_deduped = dedup_records(existing_records + new_records, gov_records)
        print(
            f"  -> {len(gov_records)} raw, {len(gov_deduped)} after dedup",
            file=sys.stderr,
        )
        new_records.extend(gov_deduped)

    # 4. Merge and fill
    print(f"Merging {len(new_records)} new windows with existing...", file=sys.stderr)
    all_records, metadata = merge_and_fill(
        existing_records, new_records, min_per_shape=args.min_per_shape,
    )

    print(json.dumps(metadata, indent=2))

    if args.dry_run:
        print(json.dumps({"status": "dry_run", "would_publish": len(all_records)}))
        return

    # 5. Publish
    from eisv_lumen.publish.hf_dataset import generate_dataset_card
    from datasets import Dataset
    from huggingface_hub import HfApi

    hf_data = trajectories_to_hf_format(all_records)
    shape_counts = dict(Counter(hf_data["shape"]))

    card = generate_dataset_card(
        dataset_name=args.repo_id,
        n_records=len(all_records),
        shape_counts=shape_counts,
    )

    dataset = Dataset.from_dict(hf_data)
    dataset.push_to_hub(args.repo_id, private=False)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )

    result = {
        "status": "published",
        "repo_id": args.repo_id,
        "n_records": len(all_records),
        "shape_counts": shape_counts,
        "new_windows_added": metadata["new_windows_added"],
        "url": f"https://huggingface.co/datasets/{args.repo_id}",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
