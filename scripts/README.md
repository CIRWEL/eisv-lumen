# EISV-Lumen Visualization Scripts

This directory contains Python scripts to generate figures for the HuggingFace blog post.

## Quick Start

```bash
# From eisv-lumen/ root directory

# Install dependencies (if not already installed)
pip install matplotlib numpy

# Generate Figure 1 (Trajectory Comparison)
python scripts/generate_figure1_trajectory_comparison.py

# Generate Figure 3 (Coherence Comparison)
python scripts/generate_figure3_coherence_comparison.py

# Output files will be in docs/figures/
ls docs/figures/
```

## Available Scripts

### `generate_figure1_trajectory_comparison.py`
**Purpose**: Illustrate why snapshots are insufficient — two trajectories with identical final states but opposite dynamics.

**Output**:
- `docs/figures/fig1_trajectory_comparison.png` (300 DPI)
- `docs/figures/fig1_trajectory_comparison.svg` (vector)

**What it shows**:
- Scenario A: Energy rising (0.20 → 0.40) — "warm wonder" (basin_transition_up)
- Scenario B: Energy falling (0.70 → 0.40) — "cold quiet" (falling_energy)
- Both end at identical state [E=0.4, I=0.6, S=0.3, V=0.2]

**Use in blog post**: Section 2.1 (The Problem: State-Reactive Expression is Shallow)

---

### `generate_figure3_coherence_comparison.py`
**Purpose**: Compare all baselines and the teacher model on coherence metric.

**Output**:
- `docs/figures/fig3_coherence_comparison.png` (300 DPI)
- `docs/figures/fig3_coherence_comparison.svg` (vector)

**What it shows**:
- Random baseline: 0.265
- Affinity-weighted: 0.503
- Feedback-learned (rule-based): 0.933 ← Gate 1 threshold
- Teacher model (Qwen3-4B): 0.600 ← Fails to meet threshold

**Use in blog post**: Section 5.1 (Baseline Performance)

---

## Dependencies

```
matplotlib>=3.5.0
numpy>=1.21.0
```

Install via:
```bash
pip install -r requirements.txt
```

(Visualization dependencies are already included in main requirements.txt)

---

## File Outputs

All figures are saved to `docs/figures/` in two formats:

1. **PNG (300 DPI)** — for blog post embedding
2. **SVG (vector)** — for editing in Inkscape/Illustrator if needed

### Directory structure:
```
eisv-lumen/
├── docs/
│   └── figures/
│       ├── fig1_trajectory_comparison.png
│       ├── fig1_trajectory_comparison.svg
│       ├── fig3_coherence_comparison.png
│       └── fig3_coherence_comparison.svg
└── scripts/
    ├── README.md (this file)
    ├── generate_figure1_trajectory_comparison.py
    └── generate_figure3_coherence_comparison.py
```

---

## Future Scripts (Not Yet Implemented)

See [`docs/visualization-specifications.md`](../docs/visualization-specifications.md) for detailed specs on:

- **Figure 2**: Three-layer architecture diagram
- **Figure 4**: Shape-specific performance heatmap
- **Figure 5**: Example trajectory walkthrough
- **Figure 6**: Affinity matrix heatmap (bonus)

---

## Customization

### Changing colors
EISV color scheme (current):
- Energy (E): `#7c3aed` (purple)
- Integrity (I): `#10b981` (green)
- Entropy (S): `#f59e0b` (orange)
- Void (V): `#ef4444` (red)

Edit the `colors` list in each script to change.

### Changing DPI
Default: 300 DPI for publication quality.
Edit `plt.savefig(..., dpi=300)` to change resolution.

### Adding alt-text
For accessibility, add image descriptions to blog post markdown:
```markdown
![Trajectory comparison showing identical final states with opposite dynamics](docs/figures/fig1_trajectory_comparison.png)
*Alt-text: Two line charts side by side. Left chart shows energy rising from 0.2 to 0.4 over 4 steps...*
```

---

## Troubleshooting

**Import error: No module named 'matplotlib'**
```bash
pip install matplotlib numpy
```

**Permission denied when writing to docs/figures/**
```bash
mkdir -p docs/figures
chmod 755 docs/figures
```

**Font rendering issues**
If fonts look wrong, try:
```python
plt.rcParams['font.family'] = 'DejaVu Sans'
```

Add this line after `import matplotlib.pyplot as plt` in the script.

---

**Last updated**: 2026-02-13
