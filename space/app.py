"""EISV-Lumen Explorer — Gradio Space for trajectory dynamics and expression generation.

Standalone app: all logic is inlined (no dependency on eisv_lumen package).
Runs on HF Spaces free tier (CPU only).
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from typing import Any, Dict, List

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants (inlined from eisv_lumen package)
# ---------------------------------------------------------------------------

ALL_TOKENS: List[str] = [
    "~warmth~", "~curiosity~", "~resonance~", "~stillness~", "~boundary~",
    "~reaching~", "~reflection~", "~ripple~", "~deep_listening~", "~emergence~",
    "~questioning~", "~holding~", "~releasing~", "~threshold~", "~return~",
]

SHAPES: List[str] = [
    "settled_presence", "convergence", "entropy_spike_recovery",
    "basin_transition_up", "basin_transition_down", "rising_entropy",
    "falling_energy", "void_rising", "drift_dissonance",
]

SHAPE_DESCRIPTIONS: Dict[str, str] = {
    "settled_presence": "Stable high-energy state with low variance",
    "convergence": "Dimensions converging toward equilibrium",
    "entropy_spike_recovery": "Sudden entropy spike followed by recovery",
    "basin_transition_up": "Sharp upward shift in energy basin",
    "basin_transition_down": "Sharp downward shift in energy basin",
    "rising_entropy": "Entropy increasing over time",
    "falling_energy": "Energy declining, withdrawal or depletion",
    "void_rising": "Void increasing as energy exceeds integrity",
    "drift_dissonance": "Sustained dissonance with no clear attractor",
}

SHAPE_TOKEN_AFFINITY: Dict[str, List[str]] = {
    "settled_presence": ["~stillness~", "~holding~", "~resonance~", "~deep_listening~"],
    "rising_entropy": ["~ripple~", "~emergence~", "~questioning~", "~curiosity~"],
    "falling_energy": ["~releasing~", "~stillness~", "~boundary~", "~reflection~"],
    "basin_transition_down": ["~releasing~", "~threshold~", "~boundary~"],
    "basin_transition_up": ["~emergence~", "~reaching~", "~warmth~", "~return~"],
    "entropy_spike_recovery": ["~ripple~", "~return~", "~holding~", "~reflection~"],
    "drift_dissonance": ["~boundary~", "~questioning~", "~reflection~"],
    "void_rising": ["~reaching~", "~curiosity~", "~questioning~", "~threshold~"],
    "convergence": ["~stillness~", "~resonance~", "~return~", "~deep_listening~"],
}

SHAPE_PATTERN_WEIGHTS: Dict[str, Dict[str, float]] = {
    "settled_presence":        {"SINGLE": 0.4, "PAIR": 0.3, "TRIPLE": 0.1, "REPETITION": 0.15, "QUESTION": 0.05},
    "rising_entropy":          {"SINGLE": 0.1, "PAIR": 0.2, "TRIPLE": 0.3, "REPETITION": 0.1, "QUESTION": 0.3},
    "falling_energy":          {"SINGLE": 0.3, "PAIR": 0.3, "TRIPLE": 0.1, "REPETITION": 0.2, "QUESTION": 0.1},
    "basin_transition_down":   {"SINGLE": 0.2, "PAIR": 0.3, "TRIPLE": 0.3, "REPETITION": 0.1, "QUESTION": 0.1},
    "basin_transition_up":     {"SINGLE": 0.15, "PAIR": 0.3, "TRIPLE": 0.35, "REPETITION": 0.1, "QUESTION": 0.1},
    "entropy_spike_recovery":  {"SINGLE": 0.1, "PAIR": 0.3, "TRIPLE": 0.3, "REPETITION": 0.2, "QUESTION": 0.1},
    "drift_dissonance":        {"SINGLE": 0.1, "PAIR": 0.2, "TRIPLE": 0.2, "REPETITION": 0.1, "QUESTION": 0.4},
    "void_rising":             {"SINGLE": 0.2, "PAIR": 0.2, "TRIPLE": 0.2, "REPETITION": 0.1, "QUESTION": 0.3},
    "convergence":             {"SINGLE": 0.4, "PAIR": 0.3, "TRIPLE": 0.1, "REPETITION": 0.15, "QUESTION": 0.05},
}

INQUIRY_TOKENS: List[str] = ["~questioning~", "~curiosity~"]

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

_dataset_cache = None


def load_dataset_samples(n: int = 200):
    """Load sample trajectories from HF dataset."""
    global _dataset_cache
    if _dataset_cache is not None:
        return _dataset_cache

    try:
        from datasets import load_dataset
        ds = load_dataset("hikewa/unitares-eisv-trajectories", split="train")
        indices = list(range(len(ds)))
        random.shuffle(indices)
        _dataset_cache = ds.select(indices[:n])
        return _dataset_cache
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None


def parse_eisv_states(eisv_json: str) -> List[Dict]:
    """Parse JSON string of EISV states into list of dicts."""
    return json.loads(eisv_json)


# ---------------------------------------------------------------------------
# Student model (JSON inference, inline copy)
# ---------------------------------------------------------------------------

_student_cache = None


class StudentInference:
    """Zero-dependency student model inference (inline copy)."""

    def __init__(self, model_dir: str):
        self._model_dir = model_dir
        self._load_models()

    def _load_models(self):
        def _load(name):
            with open(os.path.join(self._model_dir, name)) as f:
                return json.load(f)
        self._pattern_forest = _load("pattern_forest.json")
        self._token1_forest = _load("token1_forest.json")
        self._token2_forest = _load("token2_forest.json")
        self._scaler = _load("scaler.json")
        self._mappings = _load("mappings.json")

    def _scale_features(self, numeric):
        mean = self._scaler["mean"]
        scale = self._scaler["scale"]
        return [(v - m) / s for v, m, s in zip(numeric, mean, scale)]

    def _build_features(self, shape, features):
        numeric = [features.get(f, 0.0) for f in self._mappings["numeric_features"]]
        scaled = self._scale_features(numeric)
        shapes = self._mappings["shapes"]
        shape_onehot = [1.0 if s == shape else 0.0 for s in shapes]
        return scaled + shape_onehot

    def _predict_tree(self, tree, features):
        node = tree
        while not node.get("leaf", False):
            if features[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["probs"]

    def _predict_forest(self, forest, features):
        all_probs = [self._predict_tree(tree, features) for tree in forest]
        n_classes = len(all_probs[0])
        avg = [0.0] * n_classes
        for probs in all_probs:
            for i in range(n_classes):
                avg[i] += probs[i]
        best_idx = max(range(n_classes), key=lambda i: avg[i])
        return best_idx

    def predict(self, shape, features):
        X = self._build_features(shape, features)
        pattern_idx = self._predict_forest(self._pattern_forest, X)
        pattern = self._mappings["patterns"][pattern_idx]
        token1_idx = self._predict_forest(self._token1_forest, X)
        token_1 = self._mappings["tokens"][token1_idx]
        X_t2 = X + [float(token1_idx)]
        token2_idx = self._predict_forest(self._token2_forest, X_t2)
        token_2 = self._mappings["tokens_with_none"][token2_idx]

        if pattern == "SINGLE":
            tokens = [token_1]
        elif pattern == "REPETITION":
            tokens = [token_1, token_1]
        elif pattern in ("PAIR", "QUESTION"):
            tokens = [token_1, token_2] if token_2 != "none" else [token_1]
        elif pattern == "TRIPLE":
            tokens = [token_1, token_2] if token_2 != "none" else [token_1]
        else:
            tokens = [token_1]

        return {
            "pattern": pattern,
            "token_1": token_1,
            "token_2": token_2,
            "eisv_tokens": tokens,
        }


def load_student_model():
    """Download and load student model from HF Hub."""
    global _student_cache
    if _student_cache is not None:
        return _student_cache

    try:
        from huggingface_hub import hf_hub_download
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_student_cache")
        os.makedirs(model_dir, exist_ok=True)

        files = [
            "pattern_forest.json", "token1_forest.json", "token2_forest.json",
            "scaler.json", "mappings.json",
        ]
        for fname in files:
            path = os.path.join(model_dir, fname)
            if not os.path.exists(path):
                hf_hub_download(
                    "hikewa/eisv-lumen-student",
                    f"exported/{fname}",
                    local_dir=model_dir,
                    local_dir_use_symlinks=False,
                )
                # hf_hub_download may put it in a subfolder
                downloaded = os.path.join(model_dir, "exported", fname)
                if os.path.exists(downloaded) and not os.path.exists(path):
                    os.rename(downloaded, path)

        _student_cache = StudentInference(model_dir)
        return _student_cache
    except Exception as e:
        print(f"Failed to load student model: {e}")
        return None


# ---------------------------------------------------------------------------
# Rule-based expression generator (inline)
# ---------------------------------------------------------------------------

def generate_expression(shape: str) -> Dict[str, Any]:
    """Rule-based expression generation with affinity weighting."""
    patterns = SHAPE_PATTERN_WEIGHTS.get(shape, SHAPE_PATTERN_WEIGHTS["settled_presence"])
    pattern = random.choices(list(patterns.keys()), weights=list(patterns.values()), k=1)[0]

    affinity_set = set(SHAPE_TOKEN_AFFINITY.get(shape, []))
    weights = [3.0 if t in affinity_set else 1.0 for t in ALL_TOKENS]

    if pattern == "SINGLE":
        token = random.choices(ALL_TOKENS, weights=weights, k=1)[0]
        tokens = [token]
    elif pattern == "REPETITION":
        token = random.choices(ALL_TOKENS, weights=weights, k=1)[0]
        tokens = [token, token]
    elif pattern == "PAIR":
        t1, t2 = random.choices(ALL_TOKENS, weights=weights, k=2)
        tokens = [t1, t2]
    elif pattern == "TRIPLE":
        t1, t2, t3 = random.choices(ALL_TOKENS, weights=weights, k=3)
        tokens = [t1, t2, t3]
    elif pattern == "QUESTION":
        t1 = random.choices(ALL_TOKENS, weights=weights, k=1)[0]
        t2 = random.choice(INQUIRY_TOKENS)
        tokens = [t1, t2]
    else:
        tokens = [random.choices(ALL_TOKENS, weights=weights, k=1)[0]]

    # Compute coherence
    coherence = sum(1 for t in tokens if t in affinity_set) / len(tokens) if tokens else 0.0

    return {
        "pattern": pattern,
        "tokens": tokens,
        "expression": " ".join(tokens),
        "coherence": coherence,
        "affinity_tokens": sorted(affinity_set),
        "weights": {t: w for t, w in zip(ALL_TOKENS, weights)},
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_eisv_trajectory(states: List[Dict]) -> plt.Figure:
    """Plot EISV dimensions over time."""
    fig, ax = plt.subplots(figsize=(10, 5))

    n = len(states)
    x = list(range(n))

    dims = [
        ("E", "#e74c3c", "Energy"),
        ("I", "#3498db", "Info Integrity"),
        ("S", "#e67e22", "Entropy"),
        ("V", "#9b59b6", "Void"),
    ]

    for key, color, label in dims:
        values = []
        for s in states:
            if isinstance(s, dict):
                values.append(s.get(key, s.get(key.lower(), 0.0)))
            else:
                values.append(0.0)
        ax.plot(x, values, color=color, linewidth=2, label=label, alpha=0.85)

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("EISV Trajectory", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_shape_distribution(shapes: List[str]) -> plt.Figure:
    """Bar chart of shape distribution."""
    counts = Counter(shapes)
    fig, ax = plt.subplots(figsize=(10, 4))

    labels = sorted(counts.keys())
    values = [counts[l] for l in labels]
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    bars = ax.barh(labels, values, color=colors)
    ax.set_xlabel("Count", fontsize=12)
    ax.set_title("Trajectory Shape Distribution", fontsize=14, fontweight="bold")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center",
            fontsize=10,
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tab 1: Trajectory Explorer
# ---------------------------------------------------------------------------

def explore_trajectory(idx: int):
    """Load and visualize a trajectory by index."""
    ds = load_dataset_samples()
    if ds is None:
        return None, "Failed to load dataset. Please try again later."

    idx = idx % len(ds)
    row = ds[idx]

    states = parse_eisv_states(row["eisv_states"])
    shape = row["shape"]
    provenance = row["provenance"]

    fig = plot_eisv_trajectory(states)

    # Compute summary stats
    if states:
        means = {}
        for dim in ["E", "I", "S", "V"]:
            vals = [s.get(dim, s.get(dim.lower(), 0.0)) for s in states]
            means[dim] = np.mean(vals)

        info = f"**Shape**: {shape}\n"
        info += f"**Provenance**: {provenance}\n"
        info += f"**Window**: {len(states)} states\n\n"
        info += f"**Mean EISV**: E={means['E']:.3f}  I={means['I']:.3f}  S={means['S']:.3f}  V={means['V']:.3f}\n\n"
        info += f"**Description**: {SHAPE_DESCRIPTIONS.get(shape, 'Unknown shape')}"
    else:
        info = f"Shape: {shape}, no state data"

    return fig, info


def show_distribution():
    """Show shape distribution across loaded samples."""
    ds = load_dataset_samples()
    if ds is None:
        return None
    return plot_shape_distribution(ds["shape"])


# ---------------------------------------------------------------------------
# Tab 2: Expression Generator
# ---------------------------------------------------------------------------

def run_expression_generator(shape: str):
    """Generate expression for a shape and show details."""
    result = generate_expression(shape)

    output = f"### Expression: `{result['expression']}`\n\n"
    output += f"**Pattern**: {result['pattern']}\n\n"
    output += f"**Coherence**: {result['coherence']:.2f}\n\n"
    output += f"**Affinity tokens for {shape}**:\n"
    for t in result["affinity_tokens"]:
        output += f"- {t} (weight: 3.0)\n"
    output += f"\n*All other tokens have weight 1.0*\n"

    # Show pattern probabilities for this shape
    patterns = SHAPE_PATTERN_WEIGHTS.get(shape, {})
    output += f"\n**Pattern probabilities for {shape}**:\n"
    for p, w in sorted(patterns.items(), key=lambda x: -x[1]):
        bar = "\u2588" * int(w * 20)
        output += f"- {p}: {w:.0%} {bar}\n"

    return output


# ---------------------------------------------------------------------------
# Tab 3: Model Comparison
# ---------------------------------------------------------------------------

def compare_models(
    shape: str,
    mean_e: float,
    mean_i: float,
    mean_s: float,
    mean_v: float,
):
    """Compare rule-based vs student model for same input."""
    # Rule-based
    rule_result = generate_expression(shape)

    # Student
    student = load_student_model()
    features = {
        "mean_E": mean_e, "mean_I": mean_i, "mean_S": mean_s, "mean_V": mean_v,
        "dE": 0.0, "dI": 0.0, "dS": 0.0, "dV": 0.0,
        "d2E": 0.0, "d2I": 0.0, "d2S": 0.0, "d2V": 0.0,
    }

    affinity_set = set(SHAPE_TOKEN_AFFINITY.get(shape, []))

    output = "## Rule-Based (Layer 2)\n\n"
    output += f"**Expression**: `{rule_result['expression']}`\n"
    output += f"**Pattern**: {rule_result['pattern']}\n"
    output += f"**Coherence**: {rule_result['coherence']:.2f}\n\n"

    if student:
        student_result = student.predict(shape, features)
        student_tokens = student_result["eisv_tokens"]
        student_coherence = (
            sum(1 for t in student_tokens if t in affinity_set) / len(student_tokens)
            if student_tokens else 0.0
        )

        output += "## Student Model (Distilled RandomForest)\n\n"
        output += f"**Expression**: `{' '.join(student_tokens)}`\n"
        output += f"**Pattern**: {student_result['pattern']}\n"
        output += f"**Coherence**: {student_coherence:.2f}\n\n"
    else:
        output += "## Student Model\n\n"
        output += "*Student model not available yet. It will be loaded from "
        output += "[hikewa/eisv-lumen-student](https://huggingface.co/hikewa/eisv-lumen-student) "
        output += "once published.*\n\n"

    output += f"---\n\n**Affinity set for {shape}**: {', '.join(sorted(affinity_set))}\n"
    output += (
        "\n*Note: Rule-based is stochastic (click again for variation). "
        "Student is deterministic for same input.*"
    )

    return output


# ---------------------------------------------------------------------------
# Gradio App
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="EISV-Lumen Explorer",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown("""
    # EISV-Lumen Explorer

    Explore trajectory dynamics and expression generation from the
    [EISV-Lumen](https://huggingface.co/datasets/hikewa/unitares-eisv-trajectories) project.

    EISV tracks four dimensions of an embodied AI agent's state:
    **Energy** (warmth/capacity), **Information Integrity** (clarity/coherence),
    **Entropy** (uncertainty), and **Void** (disengagement).
    """)

    with gr.Tab("Trajectory Explorer"):
        gr.Markdown("Browse real trajectories from Lumen's operational history.")
        with gr.Row():
            traj_slider = gr.Slider(0, 199, value=0, step=1, label="Trajectory Index")
            traj_btn = gr.Button("Load Trajectory", variant="primary")
        traj_plot = gr.Plot(label="EISV Trajectory")
        traj_info = gr.Markdown()
        dist_btn = gr.Button("Show Shape Distribution")
        dist_plot = gr.Plot(label="Shape Distribution")

        traj_btn.click(explore_trajectory, inputs=[traj_slider], outputs=[traj_plot, traj_info])
        traj_slider.change(explore_trajectory, inputs=[traj_slider], outputs=[traj_plot, traj_info])
        dist_btn.click(show_distribution, outputs=[dist_plot])

    with gr.Tab("Expression Generator"):
        gr.Markdown(
            "Generate primitive expressions from trajectory shapes "
            "using rule-based affinity weighting."
        )
        with gr.Row():
            shape_dd = gr.Dropdown(
                choices=SHAPES, value="settled_presence", label="Trajectory Shape"
            )
            gen_btn = gr.Button("Generate Expression", variant="primary")
        gen_output = gr.Markdown()

        gen_btn.click(run_expression_generator, inputs=[shape_dd], outputs=[gen_output])

    with gr.Tab("Model Comparison"):
        gr.Markdown(
            "Compare rule-based (Layer 2) vs distilled student model output "
            "for the same trajectory input."
        )
        with gr.Row():
            cmp_shape = gr.Dropdown(
                choices=SHAPES, value="settled_presence", label="Shape"
            )
        with gr.Row():
            cmp_e = gr.Slider(0.0, 1.0, value=0.72, step=0.01, label="Mean Energy (E)")
            cmp_i = gr.Slider(0.0, 1.0, value=0.61, step=0.01, label="Mean Info Integrity (I)")
            cmp_s = gr.Slider(0.0, 1.0, value=0.18, step=0.01, label="Mean Entropy (S)")
            cmp_v = gr.Slider(0.0, 0.3, value=0.05, step=0.01, label="Mean Void (V)")
        cmp_btn = gr.Button("Compare", variant="primary")
        cmp_output = gr.Markdown()

        cmp_btn.click(
            compare_models,
            inputs=[cmp_shape, cmp_e, cmp_i, cmp_s, cmp_v],
            outputs=[cmp_output],
        )

    gr.Markdown("""
    ---
    **Models**: [Teacher (LoRA)](https://huggingface.co/hikewa/eisv-lumen-teacher) |
    [Student (RandomForest)](https://huggingface.co/hikewa/eisv-lumen-student) |
    **Dataset**: [unitares-eisv-trajectories](https://huggingface.co/datasets/hikewa/unitares-eisv-trajectories)
    """)


if __name__ == "__main__":
    demo.launch()
