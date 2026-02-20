#!/usr/bin/env python3
"""
Generate Figure 1: Trajectory Comparison
Shows why snapshots are insufficient - two trajectories with identical final states
but opposite dynamics.

Usage:
    python scripts/generate_figure1_trajectory_comparison.py

Output:
    docs/figures/fig1_trajectory_comparison.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Create output directory
output_dir = Path(__file__).parent.parent / "docs" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

# Data: Two trajectories with identical final states
# Scenario A: Rising energy (recovery) - basin_transition_up
trajectory_a = np.array([
    [0.20, 0.55, 0.28, 0.35],  # Step 1: E, I, S, V
    [0.30, 0.57, 0.29, 0.32],  # Step 2
    [0.35, 0.59, 0.30, 0.25],  # Step 3
    [0.40, 0.60, 0.30, 0.20],  # Step 4
])

# Scenario B: Falling energy (collapse) - falling_energy
trajectory_b = np.array([
    [0.70, 0.65, 0.25, 0.15],  # Step 1
    [0.60, 0.62, 0.27, 0.17],  # Step 2
    [0.50, 0.61, 0.28, 0.18],  # Step 3
    [0.40, 0.60, 0.30, 0.20],  # Step 4 (same as A!)
])

# EISV dimensions
dimensions = ['E', 'I', 'S', 'V']
colors = ['#7c3aed', '#10b981', '#f59e0b', '#ef4444']  # Purple, Green, Orange, Red
labels = ['Energy', 'Integrity', 'Entropy', 'Void']

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Steps
steps = np.arange(1, 5)

# --- Subplot 1: Scenario A (Rising) ---
for dim_idx, (color, label) in enumerate(zip(colors, labels)):
    values = trajectory_a[:, dim_idx]
    ax1.plot(steps, values, color=color, marker='o', markersize=8,
             linewidth=2.5, label=label, alpha=0.9)

# Highlight final state
ax1.axvline(x=4, linestyle='--', color='gray', alpha=0.6, linewidth=1.5)
final_state = trajectory_a[-1]
ax1.text(4.05, 0.95,
         f'Final State:\nE={final_state[0]:.2f}\nI={final_state[1]:.2f}\nS={final_state[2]:.2f}\nV={final_state[3]:.2f}',
         fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

ax1.set_title('Scenario A: Rising Energy (Recovery)\n\nShape: basin_transition_up\nExpression: "warm wonder"',
              fontsize=12, fontweight='bold', pad=15)
ax1.set_xlabel('Step', fontsize=11)
ax1.set_ylabel('EISV Value', fontsize=11)
ax1.set_ylim(-0.05, 1.05)
ax1.set_xlim(0.5, 4.5)
ax1.set_xticks(steps)
ax1.grid(True, alpha=0.3, linestyle=':')
ax1.legend(loc='upper left', framealpha=0.9)

# --- Subplot 2: Scenario B (Falling) ---
for dim_idx, (color, label) in enumerate(zip(colors, labels)):
    values = trajectory_b[:, dim_idx]
    ax2.plot(steps, values, color=color, marker='o', markersize=8,
             linewidth=2.5, label=label, alpha=0.9)

# Highlight final state
ax2.axvline(x=4, linestyle='--', color='gray', alpha=0.6, linewidth=1.5)
final_state_b = trajectory_b[-1]
ax2.text(4.05, 0.95,
         f'Final State:\nE={final_state_b[0]:.2f}\nI={final_state_b[1]:.2f}\nS={final_state_b[2]:.2f}\nV={final_state_b[3]:.2f}',
         fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

ax2.set_title('Scenario B: Falling Energy (Collapse)\n\nShape: falling_energy\nExpression: "cold quiet"',
              fontsize=12, fontweight='bold', pad=15)
ax2.set_xlabel('Step', fontsize=11)
ax2.set_ylabel('EISV Value', fontsize=11)
ax2.set_ylim(-0.05, 1.05)
ax2.set_xlim(0.5, 4.5)
ax2.set_xticks(steps)
ax2.grid(True, alpha=0.3, linestyle=':')
ax2.legend(loc='upper right', framealpha=0.9)

# Overall title
fig.suptitle('Why Snapshots Are Insufficient: Trajectory Shape Disambiguates Identical States',
             fontsize=14, fontweight='bold', y=1.02)

# Add annotation explaining the key insight
fig.text(0.5, -0.02,
         'Key insight: Both trajectories end at [E=0.4, I=0.6, S=0.3, V=0.2], but the direction of change determines expression.\n'
         'State-reactive systems would generate the same output. Trajectory-aware systems capture the dynamics.',
         ha='center', fontsize=10, style='italic', wrap=True)

plt.tight_layout()

# Save figure
output_path = output_dir / "fig1_trajectory_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Figure 1 saved to: {output_path}")

# Also save as SVG for vector editing
output_path_svg = output_dir / "fig1_trajectory_comparison.svg"
plt.savefig(output_path_svg, format='svg', bbox_inches='tight')
print(f"✅ Figure 1 (SVG) saved to: {output_path_svg}")

# Display summary
print("\nFigure 1 Summary:")
print(f"  - Scenario A: Energy rising (0.20 → 0.40), expression: 'warm wonder'")
print(f"  - Scenario B: Energy falling (0.70 → 0.40), expression: 'cold quiet'")
print(f"  - Final states: Identical [E=0.4, I=0.6, S=0.3, V=0.2]")
print(f"  - Output files: PNG (300 DPI) + SVG (vector)")
