"""
Mesh Independence Study — 3D Truss Structure (DDES)

Plots drag coefficient vs mesh cell count to demonstrate grid convergence.
Convergence is reached at ~16 million cells, with <0.1% change between
16M and 22M cell meshes.

Author: Wei Jun Yap
Project: EPSRC Vacation Internship, University of Liverpool (2024)
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# --- Load data ---
data = {"cells": [], "cd": []}
with open(os.path.join("data", "mesh_independence.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        data["cells"].append(float(row["cells_millions"]))
        data["cd"].append(float(row["drag_coefficient"]))

cells = np.array(data["cells"])
cd = np.array(data["cd"])

# --- Convergence metrics ---
deltas = np.abs(np.diff(cd) / cd[:-1]) * 100
converged_idx = np.where(deltas < 0.5)[0]  # <0.5% change

print("Mesh Independence Results")
print("-" * 50)
print(f"{'Cells (M)':>10} {'Cd':>10} {'Δ from prev':>12}")
print(f"{'':>10} {'':>10} {'(%)':>12}")
print("-" * 50)
for i, (c, d) in enumerate(zip(cells, cd)):
    delta_str = f"{deltas[i-1]:.2f}%" if i > 0 else "—"
    marker = " ← selected" if c == 16 else ""
    print(f"{c:>10.0f} {d:>10.4f} {delta_str:>12}{marker}")

final_delta = abs(cd[-1] - cd[-2]) / cd[-2] * 100
print(f"\n16M → 22M delta: {final_delta:.3f}%")

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(cells, cd, "o-", color="#2563eb", linewidth=1.5, markersize=7, 
        markerfacecolor="white", markeredgewidth=1.5, zorder=3)

# Highlight converged mesh
ax.plot(16, cd[cells == 16][0], "o", color="#dc2626", markersize=10,
        markerfacecolor="#dc2626", zorder=4, label="Selected mesh (16M)")

# Convergence band (±0.5% of final value)
cd_final = cd[-1]
ax.axhspan(cd_final * 0.995, cd_final * 1.005, alpha=0.15, color="#22c55e",
           label="±0.5% convergence band")

ax.set_xlabel("Mesh Cell Count (millions)", fontsize=12)
ax.set_ylabel("Drag Coefficient, $C_D$", fontsize=12)
ax.set_title("Mesh Independence Study — 3D Truss (DDES)", fontsize=13, fontweight="bold")
ax.legend(frameon=True, fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 24)

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig(os.path.join("figures", "mesh_independence.png"), dpi=200, bbox_inches="tight")
plt.show()
print("\nSaved: figures/mesh_independence.png")
