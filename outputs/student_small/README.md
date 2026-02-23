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

# EISV-Lumen Student -- Distilled RandomForest for Edge Deployment

A lightweight RandomForest ensemble distilled from the
[EISV-Lumen Teacher](https://huggingface.co/hikewa/eisv-lumen-teacher)
(fine-tuned Qwen2.5-0.5B). Achieves **0.986 coherence** on the EISV
expression-generation task while fitting in **~13 MB of JSON** with
**zero external dependencies** -- only Python stdlib required. Designed
to run on a Raspberry Pi 4 (Lumen's physical host).

## Model Details

| Field | Value |
|---|---|
| **Method** | Knowledge distillation (teacher-labeled soft targets) |
| **Architecture** | 3 independent RandomForest classifiers (sklearn) |
| **Input features** | 12 numeric (EISV means, deltas, accelerations) + 9 shape one-hot |
| **Training data** | 4,320 teacher-labeled examples (9 shapes x 480 each) |
| **Test data** | 1,080 held-out examples |
| **Formats** | sklearn pickle (~22 MB) and zero-dependency JSON (~13 MB) |
| **Target hardware** | Raspberry Pi 4 (1.5 GHz ARM, 4 GB RAM) |

## How It Works

The student decomposes EISV expression generation into three chained
classification problems, each solved by an independent RandomForest:

1. **Pattern classifier** -- predicts one of 5 expression patterns:
   `SINGLE`, `PAIR`, `REPETITION`, `QUESTION`, `TRIPLE`
2. **Token-1 classifier** -- predicts the primary EISV token from 15
   classes (e.g., `~stillness~`, `~warmth~`, `~emergence~`)
3. **Token-2 classifier** -- predicts the secondary token from 15 + none,
   conditioned on the Token-1 prediction (token1 index appended as extra
   feature)

The pattern determines how tokens are assembled into the final expression
string (e.g., `PAIR` yields two distinct tokens, `REPETITION` repeats
token-1 twice).

## Results

| Metric | Student (RF) | Teacher (Qwen2.5-0.5B) | Random Baseline |
|---|---|---|---|
| **Coherence** | **0.986** | 0.952 | 0.495 |
| Token-1 agreement | 0.688 | -- | -- |
| Pattern agreement | 0.652 | -- | -- |
| Full agreement (all 3 match) | 0.403 | -- | -- |

> **Why does the student exceed the teacher?** The RandomForest decision
> boundaries naturally cluster predictions toward high-affinity tokens for
> each trajectory shape. While the student disagrees with the teacher on
> exact token choices ~30% of the time, the tokens it picks are still
> coherent -- they belong to the same affinity region of EISV space. The
> coherence metric rewards any valid expression, not exact match.

## Zero-Dependency Usage (recommended for edge)

The `exported/` directory contains JSON-serialized forests and a standalone
inference module. No pip packages required.

```python
from student_inference import StudentInference

student = StudentInference("path/to/exported/")

result = student.predict("settled_presence", {
    "mean_E": 0.7, "mean_I": 0.6, "mean_S": 0.2, "mean_V": 0.05,
    "dE": 0.0, "dI": 0.0, "dS": 0.0, "dV": 0.0,
    "d2E": 0.0, "d2I": 0.0, "d2S": 0.0, "d2V": 0.0,
})
# result = {"pattern": "SINGLE", "eisv_tokens": ["~stillness~"],
#           "token_1": "~stillness~", "token_2": "none"}
```

Only `json` and `os` from the standard library are used. The inference
module walks each decision tree node-by-node and averages class
probabilities across all trees -- identical to sklearn's predict logic.

## sklearn Usage

If you have scikit-learn installed, you can use the pickle files directly:

```python
import pickle
import numpy as np

with open("pattern_clf.pkl", "rb") as f:
    pattern_clf = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("pattern_encoder.pkl", "rb") as f:
    pattern_enc = pickle.load(f)

# Build feature vector: 12 numeric features + 9 shape one-hot
numeric = np.array([[0.7, 0.6, 0.2, 0.05, 0, 0, 0, 0, 0, 0, 0, 0]])
scaled = scaler.transform(numeric)
shape_onehot = np.zeros((1, 9))  # index 7 = settled_presence
shape_onehot[0, 7] = 1.0
X = np.hstack([scaled, shape_onehot])

pattern_idx = pattern_clf.predict(X)
pattern = pattern_enc.inverse_transform(pattern_idx)[0]
```

## File Structure

```
outputs/student_small/
|-- README.md                  # This file
|-- pattern_clf.pkl            # sklearn RandomForest (4.3 MB)
|-- token1_clf.pkl             # sklearn RandomForest (8.4 MB)
|-- token2_clf.pkl             # sklearn RandomForest (9.8 MB)
|-- scaler.pkl                 # StandardScaler
|-- pattern_encoder.pkl        # LabelEncoder for patterns
|-- token1_encoder.pkl         # LabelEncoder for tokens
|-- token2_encoder.pkl         # LabelEncoder for tokens+none
|-- shape_encoder.pkl          # LabelEncoder for shapes
|-- training_metrics.json      # Cross-validation metrics
|-- eval_results.json          # Full evaluation results
|-- exported/                  # Zero-dependency JSON format
    |-- pattern_forest.json    # Decision trees as JSON (3.0 MB)
    |-- token1_forest.json     # Decision trees as JSON (4.5 MB)
    |-- token2_forest.json     # Decision trees as JSON (5.1 MB)
    |-- scaler.json            # Scaler parameters (511 B)
    |-- mappings.json          # Label mappings (1.1 KB)
    |-- student_inference.py   # Standalone inference (4.9 KB)
```

## Training Details

- **Distillation source**: Teacher (Qwen2.5-0.5B LoRA v6, 0.952 coherence
  on real Lumen trajectories)
- **Data generation**: 4,320 synthetic EISV trajectories labeled by teacher
  inference (480 per shape x 9 shapes), plus 1,080 held-out test examples
- **Forest hyperparameters**: `n_estimators=100`, `max_depth=None`,
  `random_state=42` (sklearn defaults)
- **Feature engineering**: 12 numeric features (4 EISV means + 4 first
  derivatives + 4 second derivatives) standardized via `StandardScaler`,
  plus 9-dimensional one-hot encoding of trajectory shape

## Related

- **Teacher model**: [hikewa/eisv-lumen-teacher](https://huggingface.co/hikewa/eisv-lumen-teacher) -- fine-tuned Qwen2.5-0.5B that generated the training labels
- **Dataset**: [hikewa/unitares-eisv-trajectories](https://huggingface.co/datasets/hikewa/unitares-eisv-trajectories) -- EISV trajectory data from Lumen
- **Explorer Space**: [hikewa/eisv-lumen-explorer](https://huggingface.co/spaces/hikewa/eisv-lumen-explorer) -- interactive demo

## Citation

```bibtex
@misc{eisv-lumen-student-2025,
  title   = {EISV-Lumen Student: Distilled RandomForest for Edge Deployment},
  author  = {hikewa},
  year    = {2025},
  url     = {https://huggingface.co/hikewa/eisv-lumen-student},
  note    = {Knowledge-distilled RandomForest ensemble for EISV expression
             generation on Raspberry Pi}
}
```
