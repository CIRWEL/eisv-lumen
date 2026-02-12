"""HuggingFace dataset publisher for EISV trajectory records."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


_SHAPE_DESCRIPTIONS: Dict[str, str] = {
    "settled_presence": "Stable high-energy state with low variance across EISV dimensions.",
    "rising_entropy": "Entropy (I) increasing over time, indicating growing uncertainty or exploration.",
    "falling_energy": "Energy (E) declining, signalling withdrawal or depletion.",
    "basin_transition_down": "Sharp downward shift in the energy basin (E drops across a threshold).",
    "basin_transition_up": "Sharp upward shift in the energy basin (E rises across a threshold).",
    "entropy_spike_recovery": "Sudden entropy spike followed by recovery toward baseline.",
    "drift_dissonance": "Sustained dissonance with drifting EISV values and no clear attractor.",
    "void_rising": "Low-energy void state with gradual upward movement.",
    "convergence": "Multiple EISV dimensions converging toward a shared equilibrium.",
}


def trajectories_to_hf_format(records: List[Dict[str, Any]]) -> Dict[str, List]:
    """Convert assembled trajectory records to column-oriented HF format.

    Returns a dict with these columns:
    - "shape": List[str] - shape class label
    - "eisv_states": List[str] - JSON-serialized list of state dicts
    - "derivatives": List[str] - JSON-serialized first derivatives
    - "t_start": List[float]
    - "t_end": List[float]
    - "provenance": List[str]
    - "tokens": List[str] - JSON-serialized list of token lists from expressions
    - "n_expressions": List[int] - number of aligned expressions
    """
    columns: Dict[str, List] = {
        "shape": [],
        "eisv_states": [],
        "derivatives": [],
        "t_start": [],
        "t_end": [],
        "provenance": [],
        "tokens": [],
        "n_expressions": [],
    }

    for record in records:
        columns["shape"].append(record["shape"])
        columns["eisv_states"].append(json.dumps(record["states"]))
        columns["derivatives"].append(json.dumps(record["derivatives"]))
        columns["t_start"].append(record["t_start"])
        columns["t_end"].append(record["t_end"])
        columns["provenance"].append(record["provenance"])

        expressions = record.get("expressions", [])
        token_lists = [expr["tokens"] for expr in expressions if "tokens" in expr]
        columns["tokens"].append(json.dumps(token_lists))
        columns["n_expressions"].append(len(expressions))

    return columns


def generate_dataset_card(
    dataset_name: str = "hikewa/unitares-eisv-trajectories",
    n_records: int = 0,
    shape_counts: Optional[Dict[str, int]] = None,
) -> str:
    """Generate a HuggingFace dataset card (README.md content) in markdown.

    Includes dataset name, EISV explanation, shape class descriptions,
    record counts, shape distribution, Apache 2.0 license, and citation
    placeholder.
    """
    shape_counts = shape_counts or {}

    # YAML front matter
    lines = [
        "---",
        f"license: apache-2.0",
        f"task_categories:",
        "  - text-generation",
        "tags:",
        "  - eisv",
        "  - dynamics",
        "  - trajectory",
        "size_categories:",
        f"  - {_size_category(n_records)}",
        "---",
        "",
        f"# {dataset_name}",
        "",
        "EISV trajectory dataset for dynamics-emergent voice and governance benchmarking.",
        "",
        "## EISV Framework",
        "",
        "Each trajectory tracks four continuous dimensions over time:",
        "",
        "| Dimension | Symbol | Description |",
        "|-----------|--------|-------------|",
        "| Energy    | E      | Activation level / intensity of the system |",
        "| Entropy   | I      | Uncertainty, information dispersion |",
        "| Sigma     | S      | Structural tension or dissonance |",
        "| Valence   | V      | Affective polarity (positive/negative drift) |",
        "",
        "## Trajectory Shape Classes",
        "",
        "Each record is classified into one of 9 dynamical shape classes:",
        "",
    ]

    for shape, desc in _SHAPE_DESCRIPTIONS.items():
        lines.append(f"- **{shape}**: {desc}")

    lines.append("")
    lines.append("## Dataset Statistics")
    lines.append("")
    lines.append(f"- **Total records**: {n_records}")

    if shape_counts:
        lines.append("")
        lines.append("### Shape Distribution")
        lines.append("")
        lines.append("| Shape | Count |")
        lines.append("|-------|-------|")
        for shape, count in sorted(shape_counts.items()):
            lines.append(f"| {shape} | {count} |")

    lines.extend([
        "",
        "## Schema",
        "",
        "| Column | Type | Description |",
        "|--------|------|-------------|",
        "| shape | string | Trajectory shape class label |",
        "| eisv_states | string (JSON) | Time-series of EISV state vectors |",
        "| derivatives | string (JSON) | First derivatives of EISV dimensions |",
        "| t_start | float | Start time of the trajectory window |",
        "| t_end | float | End time of the trajectory window |",
        "| provenance | string | Data source identifier |",
        "| tokens | string (JSON) | Expression token lists aligned to the trajectory |",
        "| n_expressions | int | Number of aligned expressions |",
        "",
        "## License",
        "",
        "This dataset is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).",
        "",
        "## Citation",
        "",
        "```bibtex",
        "@misc{unitares_eisv_trajectories,",
        f"  title = {{{dataset_name}}},",
        "  author = {{hikewa}},",
        "  year = {2025},",
        "  publisher = {HuggingFace},",
        "}",
        "```",
        "",
    ])

    return "\n".join(lines)


def create_hf_dataset(records: List[Dict[str, Any]]):
    """Create a HuggingFace Dataset object from assembled records.

    Uses ``datasets.Dataset.from_dict()``.
    Returns a ``datasets.Dataset`` instance.

    Note: Only call this when the ``datasets`` library is available.
    """
    from datasets import Dataset

    columns = trajectories_to_hf_format(records)
    return Dataset.from_dict(columns)


def _size_category(n: int) -> str:
    """Return HF size category string for a record count."""
    if n < 1000:
        return "n<1K"
    elif n < 10_000:
        return "1K<n<10K"
    elif n < 100_000:
        return "10K<n<100K"
    elif n < 1_000_000:
        return "100K<n<1M"
    else:
        return "n>1M"
