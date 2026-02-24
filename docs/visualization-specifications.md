# EISV-Lumen Blog Post Visualizations

## Overview

This document specifies 5 key visualizations to accompany the HuggingFace blog post. Each includes:
- **Purpose**: What it communicates
- **Data source**: Where to get the data
- **Layout**: Visual structure
- **Code**: Python/matplotlib implementation notes

---

## Figure 1: Trajectory Comparison (State vs Trajectory)

**Purpose**: Illustrate why snapshots are insufficient — two trajectories with identical final states but opposite dynamics.

**Location in post**: Section 2.1 (The Problem: State-Reactive Expression is Shallow)

**Data**:
```python
# Scenario A: Rising energy (recovery)
trajectory_a = [
    [0.2, 0.55, 0.28, 0.35],  # Step 1: E, I, S, V
    [0.3, 0.57, 0.29, 0.32],  # Step 2
    [0.35, 0.59, 0.30, 0.25], # Step 3
    [0.4, 0.60, 0.30, 0.20],  # Step 4
]
shape_a = "basin_transition_up"
expression_a = "warm wonder"

# Scenario B: Falling energy (collapse)
trajectory_b = [
    [0.7, 0.65, 0.25, 0.15],  # Step 1
    [0.6, 0.62, 0.27, 0.17],  # Step 2
    [0.5, 0.61, 0.28, 0.18],  # Step 3
    [0.4, 0.60, 0.30, 0.20],  # Step 4 (same as A!)
]
shape_b = "falling_energy"
expression_b = "cold quiet"
```

**Layout**:
- Two subplots side by side (Scenario A left, Scenario B right)
- 4 lines per subplot: E (purple), I (green), S (orange), V (red)
- X-axis: Steps 1-4
- Y-axis: EISV values [0, 1]
- Highlight final state with dashed vertical line
- Annotations:
  - "Identical final state: [E=0.4, I=0.6, S=0.3, V=0.2]"
  - Shape labels below each plot
  - Expression in primitive vocabulary

**Code outline**:
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot trajectory A
for dim, color, label in zip(range(4), ['#7c3aed', '#10b981', '#f59e0b', '#ef4444'],
                              ['E', 'I', 'S', 'V']):
    ax1.plot(range(1, 5), [step[dim] for step in trajectory_a],
             color=color, marker='o', label=label, linewidth=2)
ax1.axvline(x=4, linestyle='--', color='gray', alpha=0.5)
ax1.set_title("Scenario A: Rising Energy (Recovery)\nShape: basin_transition_up\nExpression: 'warm wonder'")
ax1.legend()
ax1.set_ylim(0, 1)
ax1.set_xlabel("Step")
ax1.set_ylabel("EISV Value")

# Repeat for trajectory B
# ...

plt.tight_layout()
plt.savefig('figure1_trajectory_comparison.png', dpi=300)
```

---

## Figure 2: Three-Layer Architecture

**Purpose**: Visual overview of the system architecture showing data flow from Layer 1 → 2 → 3.

**Location in post**: Section 3.1 (Three-Layer Architecture)

**Layout**:
```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Dataset + Benchmark                               │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐    │
│  │ 21,499      │→ │ 9 Trajectory │→ │ Coherence      │    │
│  │ Trajectories│  │ Shapes       │  │ Metric         │    │
│  └─────────────┘  └──────────────┘  └────────────────┘    │
│  Baselines: Random (0.265) | Affinity (0.503) | FB (0.933) │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Rule-Based Bridge (PRODUCTION) ✅                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐    │
│  │ Shape       │→ │ Affinity     │→ │ Expression     │    │
│  │ Classifier  │  │ Matrix 9×15  │  │ Generator      │    │
│  └─────────────┘  └──────────────┘  └────────────────┘    │
│  Coherence: 0.933 | Deployed to Raspberry Pi 4                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Fine-Tuned Teacher (RESEARCH) 🔬                  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐    │
│  │ Qwen3-4B    │→ │ LoRA Adapter │→ │ Token          │    │
│  │ Base Model  │  │ (2M params)  │  │ Sequence       │    │
│  └─────────────┘  └──────────────┘  └────────────────┘    │
│  V6 Coherence: 0.952 | Valid Rate: 100% | GATE 1: PASSED ✅  │
└─────────────────────────────────────────────────────────────┘
```

**Implementation**:
- Use matplotlib patches (Rectangle, FancyBboxPatch) for boxes
- Arrows with `FancyArrowPatch`
- Annotations for metrics
- Color scheme: Layer 1 (blue), Layer 2 (green), Layer 3 (purple)

---

## Figure 3: Coherence Comparison Chart

**Purpose**: Compare all baselines and the teacher model on coherence metric.

**Location in post**: Section 5.1 (Baseline Performance)

**Data**:
```python
models = ['Random', 'Affinity-\nWeighted', 'Feedback-\nLearned\n(Rule-Based)', 'Teacher Model\n(Qwen3-4B)']
coherence = [0.265, 0.503, 0.933, 0.952]
colors = ['#94a3b8', '#60a5fa', '#10b981', '#7c3aed']
```

**Layout**:
- Horizontal bar chart (easier to read long labels)
- Gate 1 threshold line at 0.933 (dashed red)
- Value labels at end of each bar
- Color coding: gray (random), blue (affinity), green (rule-based), purple (neural)
- Title: "Coherence Comparison: Can Neural Methods Beat Symbolic Rules?"

**Code outline**:
```python
fig, ax = plt.subplots(figsize=(10, 5))

y_pos = range(len(models))
bars = ax.barh(y_pos, coherence, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, coherence)):
    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontweight='bold')

# Gate 1 threshold
ax.axvline(x=0.933, linestyle='--', color='red', linewidth=2,
           label='Gate 1 Threshold (0.933)', alpha=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('Coherence Score', fontsize=12)
ax.set_xlim(0, 1.0)
ax.legend(loc='lower right')
ax.set_title('Coherence Comparison: Can Neural Methods Beat Symbolic Rules?',
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('figure3_coherence_comparison.png', dpi=300)
```

---

## Figure 4: Shape-Specific Performance Heatmap

**Purpose**: Show which trajectory shapes the teacher model handles well vs poorly.

**Location in post**: Section 5.5 (Shape-Specific Performance)

**Data**:
```python
shapes = [
    'settled_presence', 'rising_entropy', 'falling_energy',
    'basin_transition_down', 'basin_transition_up',
    'entropy_spike_recovery', 'drift_dissonance',
    'void_rising', 'convergence'
]
coherence = [0.993, 1.0, 0.875, 1.0, 1.0, 0.833, None, 1.0, 0.936]  # V6 real data; drift_dissonance not in eval set
valid_rate = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None, 1.0, 1.0]
```

**Layout**:
- 2 columns: Coherence | Valid Rate
- 9 rows: One per shape (note: drift_dissonance absent from V6 eval set)
- Heatmap color scale: 0.0 (red) → 1.0 (green)
- Annotate each cell with numeric value
- Highlight lowest-performing shapes (falling_energy 0.875, entropy_spike_recovery 0.833)

**Code outline**:
```python
import seaborn as sns

data = np.array([coherence, valid_rate]).T  # Shape: (9, 2)

fig, ax = plt.subplots(figsize=(6, 8))
sns.heatmap(data, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1,
            xticklabels=['Coherence', 'Valid Rate'],
            yticklabels=shapes, cbar_kws={'label': 'Score'},
            linewidths=1, linecolor='black', ax=ax)

# Highlight convergence row
ax.add_patch(plt.Rectangle((0, 8), 2, 1, fill=False, edgecolor='red',
                            linewidth=3))

ax.set_title('Teacher Model: Shape-Specific Performance',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figure4_shape_performance.png', dpi=300)
```

---

## Figure 5: Example Trajectory Walkthrough

**Purpose**: Complete example showing trajectory → shape → expression pipeline.

**Location in post**: Section 3.4 or 3.5 (Shape Classification / Expression Generation)

**Data**: Real Lumen trajectory from dataset
```python
# Example: basin_transition_up trajectory
trajectory = [
    {'E': 0.32, 'I': 0.51, 'S': 0.26, 'V': 0.38, 'time': '2025-12-01 14:23:01'},
    {'E': 0.41, 'I': 0.56, 'S': 0.27, 'V': 0.33, 'time': '2025-12-01 14:23:05'},
    {'E': 0.48, 'I': 0.61, 'S': 0.28, 'V': 0.29, 'time': '2025-12-01 14:23:09'},
    {'E': 0.55, 'I': 0.65, 'S': 0.29, 'V': 0.25, 'time': '2025-12-01 14:23:13'},
]
detected_shape = 'basin_transition_up'
eisv_tokens = ['~warmth~', '~curiosity~', '~resonance~']
lumen_primitives = ['warm', 'wonder', 'here']
```

**Layout**:
```
┌───────────────────────────────────────────────────────────┐
│  Step 1: Raw EISV Trajectory (4 steps, ~12 seconds)      │
│  [Line chart: E rising, I rising, S stable, V falling]   │
└───────────────────────────────────────────────────────────┘
                        ↓ classify_trajectory()
┌───────────────────────────────────────────────────────────┐
│  Step 2: Shape Classification                             │
│  Detected: basin_transition_up                            │
│  Reasoning: E slope > 0.15, I slope > 0.15, both positive │
└───────────────────────────────────────────────────────────┘
                        ↓ ExpressionGenerator
┌───────────────────────────────────────────────────────────┐
│  Step 3: EISV Expression Tokens                           │
│  Pattern: TRIPLE (3-token narrative)                      │
│  Tokens: ~warmth~ ~curiosity~ ~resonance~                 │
│  Affinity: [0.8, 0.7, 0.6] (high)                        │
└───────────────────────────────────────────────────────────┘
                        ↓ TOKEN_MAP translation
┌───────────────────────────────────────────────────────────┐
│  Step 4: Lumen Primitive Expression                       │
│  Output: "warm wonder here"                               │
│  Meaning: Energy rising, curiosity present, stability found│
└───────────────────────────────────────────────────────────┘
```

**Implementation**: Combine line plot + annotated flowchart

---

## Figure 6: Affinity Matrix Heatmap (Bonus)

**Purpose**: Visualize the learned shape-token relationships.

**Location in post**: Appendix or Section 3.2

**Data**: The 9×15 affinity matrix from Layer 2

**Layout**:
- Rows: 9 trajectory shapes
- Columns: 15 EISV tokens
- Heatmap color scale: 0.0 (white) → 1.0 (dark purple)
- Annotate high-affinity cells (>0.7)

---

## Implementation Priority

1. **Figure 1** (Trajectory Comparison) — Core motivation, should be first visual
2. **Figure 3** (Coherence Comparison) — Shows the gap, essential for results
3. **Figure 2** (Architecture) — Helps readers navigate the system
4. **Figure 5** (Example Walkthrough) — Makes the pipeline concrete
5. **Figure 4** (Shape Performance) — Detailed analysis for engaged readers
6. **Figure 6** (Affinity Matrix) — Nice-to-have, appendix material

---

## File Naming Convention

```
eisv-lumen/docs/figures/
├── fig1_trajectory_comparison.png
├── fig2_architecture_diagram.png
├── fig3_coherence_comparison.png
├── fig4_shape_performance_heatmap.png
├── fig5_example_walkthrough.png
└── fig6_affinity_matrix.png  (optional)
```

---

## Next Steps

1. Implement Figure 1 and Figure 3 first (highest impact)
2. Get user feedback on draft visualizations
3. Iterate on color scheme and typography to match HuggingFace blog style
4. Generate figures at 300 DPI for publication quality
5. Add alt-text descriptions for accessibility
