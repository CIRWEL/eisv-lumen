# HF Presence Phase 1 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Publish teacher model, student model, interactive Gradio Space, and blog post to Hugging Face Hub under `hikewa/`.

**Architecture:** Four independent deliverables (teacher model, student model, Space, blog post) that cross-link to each other and the existing dataset. Model uploads use `huggingface_hub` Python API. Space is a standalone Gradio app deployed as an HF Space. Blog post is the existing draft with updated links.

**Tech Stack:** huggingface_hub, peft, gradio, matplotlib, numpy, datasets

---

### Task 1: Write teacher model card

**Files:**
- Create: `outputs/teacher_lora_v6/final_adapter/README.md` (overwrite the auto-generated placeholder)

**Step 1: Write the model card**

Replace the auto-generated PEFT README with a proper model card:

```markdown
---
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3-4B
tags:
  - eisv
  - dynamics
  - trajectory
  - expression-generation
  - lora
datasets:
  - hikewa/unitares-eisv-trajectories
pipeline_tag: text-generation
---

# EISV-Lumen Teacher — LoRA on Qwen3-4B

A Qwen3-4B model fine-tuned with LoRA to generate primitive expressions from EISV trajectory dynamics. Given a trajectory shape and numeric features (means, derivatives, second derivatives of Energy, Information Integrity, Entropy, and Void), the model produces 1-3 tokens from a 15-token vocabulary that capture the trajectory's dynamical character.

## Model Details

| Property | Value |
|----------|-------|
| Base model | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) |
| Method | LoRA (rank 16, alpha 32, dropout 0.1) |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Training data | 2,880 examples (50/50 real Lumen + synthetic) |
| Training steps | 1,260 |
| Adapter size | 126 MB |

## What is EISV?

EISV is a thermodynamic framework for tracking AI agent state:

- **E** (Energy): Productive capacity, warmth — derived from CPU temperature and ambient sensors
- **I** (Information Integrity): Signal fidelity, clarity — derived from prediction accuracy and calibration
- **S** (Entropy): Semantic uncertainty — derived from environmental variance and complexity
- **V** (Void): Absence of engagement — derived from resource availability (inverse)

These are computed from real sensor readings on Lumen, a Raspberry Pi 4 with BME280 (temperature/humidity/pressure) and VEML7700 (light) sensors.

## Trajectory Shapes

The model handles 9 dynamical shape classes:

| Shape | Description | Frequency (real data) |
|-------|-------------|----------------------|
| settled_presence | Stable high-energy state | 47% |
| convergence | Dimensions converging to equilibrium | 41% |
| entropy_spike_recovery | S spike followed by recovery | 5% |
| basin_transition_up | Sharp upward energy shift | 2% |
| basin_transition_down | Sharp downward energy shift | 2% |
| rising_entropy | S increasing over time | 1.5% |
| falling_energy | E declining | 1.5% |
| void_rising | V increasing as E > I | 0.3% |
| drift_dissonance | Sustained dissonance, no attractor | synthetic only |

## Expression Vocabulary

15 primitive tokens: `~warmth~`, `~curiosity~`, `~resonance~`, `~stillness~`, `~boundary~`, `~reaching~`, `~reflection~`, `~ripple~`, `~deep_listening~`, `~emergence~`, `~questioning~`, `~holding~`, `~releasing~`, `~threshold~`, `~return~`

5 structural patterns: SINGLE, PAIR, TRIPLE, REPETITION, QUESTION

## Results

Evaluated on 500 real Lumen trajectories:

| Metric | Value |
|--------|-------|
| **Mean coherence** | **0.952** |
| Valid rate | 100% |
| Pattern accuracy | 25.8% |

Per-shape coherence:

| Shape | Coherence |
|-------|-----------|
| settled_presence | 0.993 |
| convergence | 0.936 |
| void_rising | 1.000 |
| basin_transition_down | 1.000 |
| basin_transition_up | 1.000 |
| rising_entropy | 1.000 |
| falling_energy | 0.875 |
| entropy_spike_recovery | 0.833 |

Coherence measures the fraction of generated tokens that fall within the shape's affinity set — the tokens that are contextually appropriate for that dynamical pattern.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
model = PeftModel.from_pretrained(base, "hikewa/eisv-lumen-teacher")
tokenizer = AutoTokenizer.from_pretrained("hikewa/eisv-lumen-teacher")

prompt = """<|system|>
You are an EISV expression generator. Given trajectory dynamics, output expression tokens.
<|user|>
SHAPE: settled_presence
WINDOW: n_states=20 duration=19.00
MEAN_EISV: E=0.7200 I=0.6100 S=0.1800 V=0.0500
DERIVATIVES: dE=0.0010 dI=0.0005 dS=-0.0020 dV=-0.0001
SECOND_DERIVATIVES: d2E=0.0000 d2I=0.0000 d2S=0.0001 d2V=0.0000
<|assistant|>
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Related

- **Dataset**: [hikewa/unitares-eisv-trajectories](https://huggingface.co/datasets/hikewa/unitares-eisv-trajectories) — 21,449 real trajectories from Lumen
- **Student model**: [hikewa/eisv-lumen-student](https://huggingface.co/hikewa/eisv-lumen-student) — Distilled RandomForest for Pi deployment (0.986 coherence)
- **Interactive demo**: [hikewa/eisv-lumen-explorer](https://huggingface.co/spaces/hikewa/eisv-lumen-explorer)

## Citation

```bibtex
@misc{eisv_lumen_teacher_2026,
  title = {EISV-Lumen Teacher: LoRA on Qwen3-4B for Trajectory-Aware Expression},
  author = {hikewa},
  year = {2026},
  publisher = {HuggingFace},
  url = {https://huggingface.co/hikewa/eisv-lumen-teacher},
}
```
```

**Step 2: Verify the model card renders correctly**

Run: `python3 -c "open('outputs/teacher_lora_v6/final_adapter/README.md').read()"`
Expected: No errors, content matches above.

**Step 3: Commit**

```bash
git add outputs/teacher_lora_v6/final_adapter/README.md
git commit -m "Write teacher model card for HF upload"
```

---

### Task 2: Upload teacher model to HF

**Files:**
- Create: `eisv_lumen/scripts/publish_model.py`

**Step 1: Write the upload script**

```python
"""Upload EISV-Lumen models to HuggingFace Hub.

Usage:
    python3 -m eisv_lumen.scripts.publish_model teacher [--dry-run]
    python3 -m eisv_lumen.scripts.publish_model student [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TEACHER_REPO = "hikewa/eisv-lumen-teacher"
STUDENT_REPO = "hikewa/eisv-lumen-student"

TEACHER_DIR = Path(__file__).resolve().parents[2] / "outputs" / "teacher_lora_v6" / "final_adapter"
STUDENT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "student_small"

# Files to upload for teacher (LoRA adapter + tokenizer)
TEACHER_FILES = [
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "README.md",
]

# Files to upload for student (exported JSON models + inference script)
STUDENT_FILES_EXPORTED = [
    "exported/pattern_forest.json",
    "exported/token1_forest.json",
    "exported/token2_forest.json",
    "exported/scaler.json",
    "exported/mappings.json",
    "exported/student_inference.py",
]

STUDENT_FILES_ROOT = [
    "eval_results.json",
    "training_metrics.json",
    "README.md",
]


def upload_teacher(dry_run: bool = False) -> None:
    """Upload teacher LoRA adapter to HF Hub."""
    print(f"Uploading teacher model to {TEACHER_REPO}")
    print(f"Source: {TEACHER_DIR}")

    # Validate files exist
    for fname in TEACHER_FILES:
        path = TEACHER_DIR / fname
        if not path.exists():
            print(f"  WARNING: {fname} not found, skipping", file=sys.stderr)
        else:
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"  {fname}: {size_mb:.1f} MB")

    if dry_run:
        print("DRY RUN — would upload above files")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(TEACHER_REPO, exist_ok=True)

    for fname in TEACHER_FILES:
        path = TEACHER_DIR / fname
        if path.exists():
            print(f"  Uploading {fname}...")
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=fname,
                repo_id=TEACHER_REPO,
            )

    print(f"Done: https://huggingface.co/{TEACHER_REPO}")


def upload_student(dry_run: bool = False) -> None:
    """Upload student model to HF Hub."""
    print(f"Uploading student model to {STUDENT_REPO}")
    print(f"Source: {STUDENT_DIR}")

    all_files = STUDENT_FILES_EXPORTED + STUDENT_FILES_ROOT
    for fname in all_files:
        path = STUDENT_DIR / fname
        if not path.exists():
            print(f"  WARNING: {fname} not found, skipping", file=sys.stderr)
        else:
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"  {fname}: {size_mb:.1f} MB")

    if dry_run:
        print("DRY RUN — would upload above files")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(STUDENT_REPO, exist_ok=True)

    for fname in all_files:
        path = STUDENT_DIR / fname
        if path.exists():
            print(f"  Uploading {fname}...")
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=fname,
                repo_id=STUDENT_REPO,
            )

    print(f"Done: https://huggingface.co/{STUDENT_REPO}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload EISV-Lumen models to HF Hub")
    parser.add_argument("model", choices=["teacher", "student"], help="Which model to upload")
    parser.add_argument("--dry-run", action="store_true", help="Validate without uploading")
    args = parser.parse_args()

    if args.model == "teacher":
        upload_teacher(dry_run=args.dry_run)
    else:
        upload_student(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
```

**Step 2: Dry-run to validate**

Run: `cd /Users/cirwel/projects/eisv-lumen && python3 -m eisv_lumen.scripts.publish_model teacher --dry-run`
Expected: Lists all files with sizes, prints "DRY RUN"

**Step 3: Upload for real**

Run: `cd /Users/cirwel/projects/eisv-lumen && python3 -m eisv_lumen.scripts.publish_model teacher`
Expected: All files uploaded, prints URL

**Step 4: Verify on HF**

Check: `https://huggingface.co/hikewa/eisv-lumen-teacher`
Expected: Model card renders, adapter files visible, tagged as PEFT/LoRA

**Step 5: Commit**

```bash
git add eisv_lumen/scripts/publish_model.py
git commit -m "Add model upload script for HF Hub"
```

---

### Task 3: Write student model card

**Files:**
- Create: `outputs/student_small/README.md`

**Step 1: Write the model card**

```markdown
---
license: apache-2.0
tags:
  - eisv
  - dynamics
  - trajectory
  - expression-generation
  - random-forest
  - edge-deployment
  - raspberry-pi
datasets:
  - hikewa/unitares-eisv-trajectories
---

# EISV-Lumen Student — Distilled RandomForest for Edge Deployment

A lightweight RandomForest ensemble distilled from the [EISV-Lumen Teacher](https://huggingface.co/hikewa/eisv-lumen-teacher) for deployment on Raspberry Pi 4. Achieves **0.986 coherence** (vs teacher's 0.952) in a **~13 MB JSON** package with zero external dependencies.

## Model Details

| Property | Value |
|----------|-------|
| Method | Knowledge distillation: teacher labels → RandomForest |
| Architecture | 3 independent classifiers (pattern, token1, token2) |
| Input features | 12 numeric (EISV means + derivatives + 2nd derivatives) + shape one-hot |
| Training data | 4,320 teacher-labeled examples |
| Formats | sklearn pickle (~35 MB) and zero-dependency JSON (~13 MB) |

## How It Works

The student consists of three RandomForest classifiers chained together:

1. **Pattern classifier** → predicts structural pattern (SINGLE, PAIR, TRIPLE, REPETITION, QUESTION)
2. **Token-1 classifier** → predicts primary token from 15-token vocabulary
3. **Token-2 classifier** → predicts secondary token (or "none"), conditioned on token-1

Input features are standardized (mean/variance normalization) and combined with a shape one-hot encoding.

## Results

| Metric | Student (this) | Teacher (LoRA) | Baseline (random) |
|--------|---------------|----------------|-------------------|
| **Mean coherence** | **0.986** | 0.952 | 0.495 |
| Token-1 agreement | 0.688 | — | — |
| Pattern agreement | 0.652 | — | — |
| Full agreement | 0.403 | — | — |
| Inference device | Raspberry Pi 4 | GPU | CPU |
| Size | 13 MB (JSON) | 126 MB (LoRA) | 0 |

The student exceeds the teacher in coherence because the RandomForest's decision boundaries naturally cluster toward high-affinity tokens, while the teacher's stochastic generation occasionally samples outside the affinity set.

## Usage — Zero-Dependency (Recommended for Edge)

```python
from student_inference import StudentInference

student = StudentInference("./exported/")

result = student.predict("settled_presence", {
    "mean_E": 0.72, "mean_I": 0.61, "mean_S": 0.18, "mean_V": 0.05,
    "dE": 0.001, "dI": 0.0005, "dS": -0.002, "dV": -0.0001,
    "d2E": 0.0, "d2I": 0.0, "d2S": 0.0001, "d2V": 0.0,
})

print(result)
# {"pattern": "SINGLE", "token_1": "~stillness~", "token_2": "none", "eisv_tokens": ["~stillness~"]}
```

Requires only Python stdlib (json, os). No numpy, sklearn, or any other dependency.

## Usage — sklearn

```python
import joblib
import numpy as np

pattern_clf = joblib.load("pattern_clf.pkl")
token1_clf = joblib.load("token1_clf.pkl")
token2_clf = joblib.load("token2_clf.pkl")
scaler = joblib.load("scaler.pkl")
# ... see full example in student_inference.py
```

## File Structure

```
├── exported/                    # Zero-dependency JSON format
│   ├── pattern_forest.json     # Pattern classifier (3.0 MB)
│   ├── token1_forest.json      # Token-1 classifier (4.7 MB)
│   ├── token2_forest.json      # Token-2 classifier (5.3 MB)
│   ├── scaler.json             # StandardScaler params
│   ├── mappings.json           # Class labels and feature names
│   └── student_inference.py    # Standalone inference module
├── eval_results.json           # Evaluation metrics
└── training_metrics.json       # Training metrics
```

## Related

- **Teacher model**: [hikewa/eisv-lumen-teacher](https://huggingface.co/hikewa/eisv-lumen-teacher) — LoRA on Qwen3-4B (0.952 coherence)
- **Dataset**: [hikewa/unitares-eisv-trajectories](https://huggingface.co/datasets/hikewa/unitares-eisv-trajectories) — 21,449 real trajectories
- **Interactive demo**: [hikewa/eisv-lumen-explorer](https://huggingface.co/spaces/hikewa/eisv-lumen-explorer)

## Citation

```bibtex
@misc{eisv_lumen_student_2026,
  title = {EISV-Lumen Student: Distilled RandomForest for Edge Expression Generation},
  author = {hikewa},
  year = {2026},
  publisher = {HuggingFace},
  url = {https://huggingface.co/hikewa/eisv-lumen-student},
}
```
```

**Step 2: Commit**

```bash
git add outputs/student_small/README.md
git commit -m "Write student model card for HF upload"
```

---

### Task 4: Upload student model to HF

**Step 1: Dry-run to validate**

Run: `cd /Users/cirwel/projects/eisv-lumen && python3 -m eisv_lumen.scripts.publish_model student --dry-run`
Expected: Lists all files with sizes, prints "DRY RUN"

**Step 2: Upload for real**

Run: `cd /Users/cirwel/projects/eisv-lumen && python3 -m eisv_lumen.scripts.publish_model student`
Expected: All files uploaded, prints URL

**Step 3: Verify on HF**

Check: `https://huggingface.co/hikewa/eisv-lumen-student`
Expected: Model card renders, JSON and pkl files visible

---

### Task 5: Build the Gradio Space app

**Files:**
- Create: `space/app.py`
- Create: `space/requirements.txt`
- Create: `space/README.md` (Space metadata)

**Step 1: Create the Space directory**

```bash
mkdir -p /Users/cirwel/projects/eisv-lumen/space
```

**Step 2: Write `space/README.md`** (HF Space metadata)

```markdown
---
title: EISV-Lumen Explorer
emoji: 🌊
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
pinned: false
license: apache-2.0
datasets:
  - hikewa/unitares-eisv-trajectories
models:
  - hikewa/eisv-lumen-teacher
  - hikewa/eisv-lumen-student
tags:
  - eisv
  - dynamics
  - trajectory
  - expression-generation
---

# EISV-Lumen Explorer

Interactive demo for exploring EISV trajectory dynamics and expression generation.

- **Trajectory Explorer**: Visualize EISV time-series and trajectory shape classification
- **Expression Generator**: Generate primitive expressions from trajectory shapes using rule-based affinity weighting
- **Model Comparison**: Compare rule-based vs distilled student model outputs
```

**Step 3: Write `space/requirements.txt`**

```
gradio>=5.0
matplotlib
numpy
datasets
huggingface_hub
```

**Step 4: Write `space/app.py`**

The full Gradio app with three tabs. This is the main deliverable — see implementation below.

The app must:
1. Load sample trajectories from `hikewa/unitares-eisv-trajectories` on startup
2. Plot EISV dimensions over time with matplotlib
3. Include the expression generator logic inline (no dependency on eisv_lumen package)
4. Include the student JSON inference inline (download from `hikewa/eisv-lumen-student`)
5. Run on free HF Space tier (CPU only, no GPU)

```python
"""EISV-Lumen Explorer — Interactive Gradio Space.

Explore EISV trajectory dynamics, generate expressions, and compare models.
Runs on CPU. No GPU required.
"""

import json
import os
import random
from typing import Any, Dict, List, Tuple

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_TOKENS = [
    "~warmth~", "~curiosity~", "~resonance~", "~stillness~", "~boundary~",
    "~reaching~", "~reflection~", "~ripple~", "~deep_listening~", "~emergence~",
    "~questioning~", "~holding~", "~releasing~", "~threshold~", "~return~",
]

SHAPES = [
    "settled_presence", "convergence", "entropy_spike_recovery",
    "basin_transition_up", "basin_transition_down", "rising_entropy",
    "falling_energy", "void_rising", "drift_dissonance",
]

SHAPE_DESCRIPTIONS = {
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

SHAPE_TOKEN_AFFINITY = {
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

SHAPE_PATTERN_WEIGHTS = {
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

INQUIRY_TOKENS = ["~questioning~", "~curiosity~"]

# ---------------------------------------------------------------------------
# Data loading
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
        # Sample up to n records, stratified by shape
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
# Student model (JSON inference, inline)
# ---------------------------------------------------------------------------

_student_cache = None


def load_student_model():
    """Download and load student model from HF Hub."""
    global _student_cache
    if _student_cache is not None:
        return _student_cache

    try:
        from huggingface_hub import hf_hub_download
        model_dir = os.path.join(os.path.dirname(__file__), "_student_cache")
        os.makedirs(model_dir, exist_ok=True)

        files = ["pattern_forest.json", "token1_forest.json", "token2_forest.json",
                 "scaler.json", "mappings.json"]
        for fname in files:
            path = os.path.join(model_dir, fname)
            if not os.path.exists(path):
                hf_hub_download("hikewa/eisv-lumen-student", f"exported/{fname}",
                                local_dir=model_dir, local_dir_use_symlinks=False)
                # hf_hub_download may put it in a subfolder
                downloaded = os.path.join(model_dir, "exported", fname)
                if os.path.exists(downloaded) and not os.path.exists(path):
                    os.rename(downloaded, path)

        _student_cache = StudentInference(model_dir)
        return _student_cache
    except Exception as e:
        print(f"Failed to load student model: {e}")
        return None


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

        return {"pattern": pattern, "token_1": token_1, "token_2": token_2, "eisv_tokens": tokens}


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
    from collections import Counter
    counts = Counter(shapes)
    fig, ax = plt.subplots(figsize=(10, 4))

    labels = sorted(counts.keys())
    values = [counts[l] for l in labels]
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    bars = ax.barh(labels, values, color=colors)
    ax.set_xlabel("Count", fontsize=12)
    ax.set_title("Trajectory Shape Distribution", fontsize=14, fontweight="bold")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=10)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tab 1: Trajectory Explorer
# ---------------------------------------------------------------------------

def explore_trajectory(idx: int):
    """Load and visualize a trajectory by index."""
    ds = load_dataset_samples()
    if ds is None:
        return None, "Failed to load dataset", ""

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
        bar = "█" * int(w * 20)
        output += f"- {p}: {w:.0%} {bar}\n"

    return output


# ---------------------------------------------------------------------------
# Tab 3: Model Comparison
# ---------------------------------------------------------------------------

def compare_models(shape: str, mean_e: float, mean_i: float, mean_s: float, mean_v: float):
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
        student_coherence = sum(1 for t in student_tokens if t in affinity_set) / len(student_tokens) if student_tokens else 0.0

        output += "## Student Model (Distilled RandomForest)\n\n"
        output += f"**Expression**: `{' '.join(student_tokens)}`\n"
        output += f"**Pattern**: {student_result['pattern']}\n"
        output += f"**Coherence**: {student_coherence:.2f}\n\n"
    else:
        output += "## Student Model\n\n*Failed to load student model*\n\n"

    output += f"---\n\n**Affinity set for {shape}**: {', '.join(sorted(affinity_set))}\n"
    output += f"\n*Note: Rule-based is stochastic (click again for variation). Student is deterministic for same input.*"

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
        gr.Markdown("Generate primitive expressions from trajectory shapes using rule-based affinity weighting.")
        with gr.Row():
            shape_dd = gr.Dropdown(choices=SHAPES, value="settled_presence", label="Trajectory Shape")
            gen_btn = gr.Button("Generate Expression", variant="primary")
        gen_output = gr.Markdown()

        gen_btn.click(run_expression_generator, inputs=[shape_dd], outputs=[gen_output])

    with gr.Tab("Model Comparison"):
        gr.Markdown("Compare rule-based (Layer 2) vs distilled student model output for the same trajectory input.")
        with gr.Row():
            cmp_shape = gr.Dropdown(choices=SHAPES, value="settled_presence", label="Shape")
        with gr.Row():
            cmp_e = gr.Slider(0.0, 1.0, value=0.72, step=0.01, label="Mean Energy (E)")
            cmp_i = gr.Slider(0.0, 1.0, value=0.61, step=0.01, label="Mean Info Integrity (I)")
            cmp_s = gr.Slider(0.0, 1.0, value=0.18, step=0.01, label="Mean Entropy (S)")
            cmp_v = gr.Slider(0.0, 0.3, value=0.05, step=0.01, label="Mean Void (V)")
        cmp_btn = gr.Button("Compare", variant="primary")
        cmp_output = gr.Markdown()

        cmp_btn.click(compare_models, inputs=[cmp_shape, cmp_e, cmp_i, cmp_s, cmp_v], outputs=[cmp_output])

    gr.Markdown("""
    ---
    **Models**: [Teacher (LoRA)](https://huggingface.co/hikewa/eisv-lumen-teacher) |
    [Student (RandomForest)](https://huggingface.co/hikewa/eisv-lumen-student) |
    **Dataset**: [unitares-eisv-trajectories](https://huggingface.co/datasets/hikewa/unitares-eisv-trajectories)
    """)


if __name__ == "__main__":
    demo.launch()
```

**Step 5: Test locally**

Run: `cd /Users/cirwel/projects/eisv-lumen/space && python3 app.py`
Expected: Gradio launches on localhost, all three tabs work.

**Step 6: Commit**

```bash
git add space/
git commit -m "Add Gradio Space for EISV-Lumen Explorer"
```

---

### Task 6: Deploy Space to HF

**Step 1: Create the Space repo and push**

```bash
cd /Users/cirwel/projects/eisv-lumen/space
git init
git remote add origin https://huggingface.co/spaces/hikewa/eisv-lumen-explorer
git add .
git commit -m "Initial Space: EISV-Lumen Explorer"
git push -u origin main
```

Alternatively, use HfApi:

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("hikewa/eisv-lumen-explorer", repo_type="space", space_sdk="gradio", exist_ok=True)
api.upload_folder(folder_path="space/", repo_id="hikewa/eisv-lumen-explorer", repo_type="space")
```

**Step 2: Verify Space loads**

Check: `https://huggingface.co/spaces/hikewa/eisv-lumen-explorer`
Expected: Gradio app loads, all three tabs functional.

---

### Task 7: Update dataset card with cross-links

**Step 1: Update the dataset card generator to include links to models and Space**

**Files:**
- Modify: `eisv_lumen/publish/hf_dataset.py` — add model/Space links in the "Models Trained" section

Add after the existing models section in `generate_dataset_card()`:

```python
# After the student table, add:
"",
"### Links",
"",
"- **Teacher model**: [hikewa/eisv-lumen-teacher](https://huggingface.co/hikewa/eisv-lumen-teacher)",
"- **Student model**: [hikewa/eisv-lumen-student](https://huggingface.co/hikewa/eisv-lumen-student)",
"- **Interactive demo**: [EISV-Lumen Explorer](https://huggingface.co/spaces/hikewa/eisv-lumen-explorer)",
```

**Step 2: Re-publish the dataset card**

Run: `cd /Users/cirwel/projects/eisv-lumen && python3 -c "
from eisv_lumen.publish.hf_dataset import generate_dataset_card
from huggingface_hub import HfApi
card = generate_dataset_card('hikewa/unitares-eisv-trajectories', n_records=21499, shape_counts={})
api = HfApi()
api.upload_file(path_or_fileobj=card.encode(), path_in_repo='README.md', repo_id='hikewa/unitares-eisv-trajectories', repo_type='dataset')
"`

**Step 3: Commit**

```bash
git add eisv_lumen/publish/hf_dataset.py
git commit -m "Add model and Space cross-links to dataset card"
```

---

### Task 8: Review and update blog post

**Files:**
- Modify: `docs/huggingface-blog-post-draft.md`

**Step 1: Review the blog post for outdated metrics**

Read through and verify all numbers match V6 eval results:
- Mean coherence: 0.952 (teacher), 0.986 (student-small), 0.933 (rule-based)
- Dataset size: 21,449 real + 50 synthetic = 21,499
- Training: 2,880 examples, 1,260 steps

**Step 2: Add links to newly published HF assets**

Add near the top and in relevant sections:
- Teacher model: `https://huggingface.co/hikewa/eisv-lumen-teacher`
- Student model: `https://huggingface.co/hikewa/eisv-lumen-student`
- Interactive demo: `https://huggingface.co/spaces/hikewa/eisv-lumen-explorer`

**Step 3: Commit**

```bash
git add docs/huggingface-blog-post-draft.md
git commit -m "Update blog post with HF model and Space links"
```

**Step 4: Submit blog post**

HF blog submission is done via PR to `huggingface/blog` repo on GitHub. Create a PR with the markdown file.

Run:
```bash
# Fork huggingface/blog, add the post
gh repo fork huggingface/blog --clone
cp /Users/cirwel/projects/eisv-lumen/docs/huggingface-blog-post-draft.md blog/eisv-lumen-trajectory-expression.md
cd blog
git checkout -b eisv-lumen-blog-post
git add eisv-lumen-trajectory-expression.md
git commit -m "Add EISV-Lumen blog post: trajectory-aware expression for embodied AI"
git push -u origin eisv-lumen-blog-post
gh pr create --title "Blog post: EISV-Lumen trajectory-aware expression" --body "Blog post about EISV-Lumen, a three-layer system for trajectory-aware expression in embodied AI."
```

---

### Task 9: Final verification

**Step 1: Verify all cross-links work**

Check each URL resolves:
- `https://huggingface.co/datasets/hikewa/unitares-eisv-trajectories`
- `https://huggingface.co/hikewa/eisv-lumen-teacher`
- `https://huggingface.co/hikewa/eisv-lumen-student`
- `https://huggingface.co/spaces/hikewa/eisv-lumen-explorer`

**Step 2: Verify model cards render correctly**

On each model page, check:
- YAML metadata shows correct tags/datasets/license
- Usage examples have correct repo IDs
- Related links section points to other assets

**Step 3: Verify Space is functional**

- Tab 1: Load a trajectory, see EISV plot and shape info
- Tab 2: Select each shape, generate expressions, see coherence scores
- Tab 3: Compare rule-based vs student output

**Step 4: Final commit in eisv-lumen repo**

```bash
git add -A
git commit -m "Complete HF presence phase 1: models, Space, blog post"
```
