"""HuggingFace dataset publisher for EISV trajectory records."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


_SHAPE_DESCRIPTIONS: Dict[str, str] = {
    "settled_presence": "Stable high-energy state with low variance across EISV dimensions.",
    "rising_entropy": "Entropy (S) increasing over time, indicating growing uncertainty or exploration.",
    "falling_energy": "Energy (E) declining, signalling withdrawal or depletion.",
    "basin_transition_down": "Sharp downward shift in the energy basin (E and I drop across a threshold).",
    "basin_transition_up": "Sharp upward shift in the energy basin (E and I rise across a threshold).",
    "entropy_spike_recovery": "Sudden entropy (S) spike followed by recovery toward baseline.",
    "drift_dissonance": "Sustained dissonance with drifting EISV values and no clear attractor.",
    "void_rising": "V increasing as energy exceeds integrity (E > I); presence fading.",
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
        "**Source**: [CIRWEL/eisv-lumen](https://github.com/CIRWEL/eisv-lumen)",
        "",
        "## EISV Framework",
        "",
        "Each trajectory tracks four continuous dimensions over time:",
        "",
        "| Dimension | Symbol | Range | Description |",
        "|-----------|--------|-------|-------------|",
        "| Energy | E | [0, 1] | Productive capacity; couples toward I via α(I−E), reduced by entropy cross-coupling |",
        "| Information Integrity | I | [0, 1] | Signal fidelity; boosted by coherence C(V,Θ), reduced by entropy |",
        "| Entropy | S | [0, 1] | Semantic uncertainty; decays naturally, rises with complexity and drift |",
        "| Void | V | [0, 0.3] | Absence of engagement; V = (1 − presence) × 0.3 |",
        "",
        "**Note on ranges**: These are observation-layer values from Lumen's sensors. The UNITARES governance ODE evolves S to [0, 2] and V to [−2, 2] as a signed E−I imbalance integrator, but the raw trajectories in this dataset use the sensor ranges above.",
        "",
        "### What is Lumen?",
        "",
        "Lumen is a Raspberry Pi with environmental sensors (BME280 for temperature/humidity/pressure, VEML7700 for light) that maps physical readings to EISV dimensions: warmth→E, clarity→I, (1−stability)→S, (1−presence)×0.3→V. The trajectories in this dataset are time-windowed snapshots of these sensor-derived EISV states.",
        "",
        "## Trajectory Shape Classes",
        "",
        "Each record is classified into one of 9 dynamical shape classes:",
        "",
    ]

    for shape, desc in _SHAPE_DESCRIPTIONS.items():
        lines.append(f"- **{shape}**: {desc}")

    lines.extend([
        "",
        "### Shape Classification Note",
        "",
        "Shape labels in this dataset were generated from **20-step trajectory windows**.",
        "If you reclassify using shorter windows, expect significant label disagreement:",
        "",
        "| Window Size | Label Match |",
        "|-------------|-------------|",
        "| 4-step | 65% |",
        "| 8-step | 77% |",
        "| 10-step | 81% |",
        "| 15-step | 91% |",
        "| 20-step | 100% |",
        "",
        "Most mismatches (5,138 cases) are `settled_presence` → `convergence`.",
        "A 4-step window only sees the tail end of a settling trajectory, which looks",
        "like convergence. The full 20-step arc is needed to confirm the system has",
        "actually settled. If training on this dataset, use at least 10-15 steps for",
        "reliable shape classification.",
        "",
    ])

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
        "| provenance | string | `lumen_real` (from Lumen sensors) or `synthetic` (generated to fill underrepresented shapes) |",
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
        "  year = {2026},",
        "  publisher = {HuggingFace},",
        f"  url = {{https://huggingface.co/datasets/{dataset_name}}},",
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
