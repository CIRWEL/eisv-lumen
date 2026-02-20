#!/usr/bin/env python3
"""
Generate Figure 3: Coherence Comparison
Bar chart comparing all baselines and the teacher model on coherence metric.
Shows the 0.60 vs 0.933 gap with Gate 1 threshold.

Usage:
    python scripts/generate_figure3_coherence_comparison.py

Output:
    docs/figures/fig3_coherence_comparison.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Create output directory
output_dir = Path(__file__).parent.parent / "docs" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

# Data
models = [
    'Random\nBaseline',
    'Affinity-Weighted\nBaseline',
    'Feedback-Learned\nRule-Based\n(Layer 2)',
    'Teacher Model\nQwen3-4B LoRA\n(Layer 3)'
]
coherence = [0.265, 0.503, 0.933, 0.600]
colors = ['#94a3b8', '#60a5fa', '#10b981', '#7c3aed']  # Gray, Blue, Green, Purple
labels = ['Random', 'Affinity', 'Rule-Based (Production ✅)', 'Neural (Research 🔬)']

# Gate 1 threshold
gate1_threshold = 0.933

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

# Horizontal bar chart
y_pos = np.arange(len(models))
bars = ax.barh(y_pos, coherence, color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)

# Add value labels at end of each bar
for i, (bar, val, color) in enumerate(zip(bars, coherence, colors)):
    ax.text(val + 0.015, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontweight='bold', fontsize=12)

    # Add percentage of rule-based performance
    if i == 3:  # Teacher model
        pct = (val / gate1_threshold) * 100
        ax.text(val + 0.015, bar.get_y() + bar.get_height()/2 - 0.15,
                f'({pct:.1f}% of rule-based)', va='center', fontsize=9, style='italic', color='gray')

# Gate 1 threshold line
ax.axvline(x=gate1_threshold, linestyle='--', color='red', linewidth=2.5,
           label=f'Gate 1 Threshold ({gate1_threshold:.3f})', alpha=0.8, zorder=0)

# Add threshold annotation
ax.text(gate1_threshold + 0.01, 3.5,
        'Gate 1: coherence > 0.933\nAND valid rate > 90%',
        fontsize=9, color='red', verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.7))

# Styling
ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=11)
ax.set_xlabel('Coherence Score', fontsize=12, fontweight='bold')
ax.set_xlim(0, 1.0)
ax.set_xticks(np.arange(0, 1.1, 0.1))
ax.grid(True, axis='x', alpha=0.3, linestyle=':')
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

# Title
ax.set_title('Coherence Comparison: Can Neural Methods Beat Symbolic Rules?\n'
             'The 0.60 → 0.933 Gap Reveals Trajectory-Aware Expression Challenges',
             fontsize=14, fontweight='bold', pad=20)

# Add insights box
insights_text = (
    "Key Findings:\n"
    "• Domain structure (affinity matrix) accounts for ~50% of task (0.265 → 0.503)\n"
    "• Feedback learning accounts for ~90% improvement (0.503 → 0.933)\n"
    "• Neural model (4B params) achieves 64% of rule-based performance\n"
    "• Gap suggests symbolic structure may be essential for this task"
)
ax.text(0.02, 0.5, insights_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='gray'))

plt.tight_layout()

# Save figure
output_path = output_dir / "fig3_coherence_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Figure 3 saved to: {output_path}")

# Also save as SVG
output_path_svg = output_dir / "fig3_coherence_comparison.svg"
plt.savefig(output_path_svg, format='svg', bbox_inches='tight')
print(f"✅ Figure 3 (SVG) saved to: {output_path_svg}")

# Display summary
print("\nFigure 3 Summary:")
print(f"  - Random baseline: {coherence[0]:.3f} (zero learning)")
print(f"  - Affinity-weighted: {coherence[1]:.3f} (domain structure only)")
print(f"  - Feedback-learned: {coherence[2]:.3f} (rule-based, deployed)")
print(f"  - Teacher model: {coherence[3]:.3f} ({(coherence[3]/coherence[2])*100:.1f}% of rule-based)")
print(f"  - Gate 1 threshold: {gate1_threshold:.3f} (NOT MET ❌)")
print(f"  - Output files: PNG (300 DPI) + SVG (vector)")
