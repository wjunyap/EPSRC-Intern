"""
Drag Coefficient Comparison — LES vs DDES vs RANS vs Porous Volume

Compares drag coefficients of a 3D truss structure across four modelling
approaches. Demonstrates that porous-zone substitution underpredicts drag
by ~40% relative to DDES, making it unsuitable for safety-critical
applications such as CAP 437 helideck turbulence assessment.

Author: Wei Jun Yap
Project: EPSRC Vacation Internship, University of Liverpool (2024)
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# --- Load data ---
methods = []
cd_values = []
sources = []
with open(os.path.join("data", "drag_coefficients.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        methods.append(row["method"])
        cd_values.append(float(row["drag_coefficient"]))
        sources.append(row["source"])

cd = np.array(cd_values)

# --- Compute differences ---
les_cd = cd[methods.index("LES")]
ddes_cd = cd[methods.index("DDES")]

print("3D Truss Drag Coefficient Comparison")
print("=" * 60)
print(f"{'Method':<16} {'Cd':>6} {'vs LES':>10} {'vs DDES':>10}  Source")
print("-" * 60)
for m, c, s in zip(methods, cd, sources):
    diff_les = (c - les_cd) / les_cd * 100
    diff_ddes = (c - ddes_cd) / ddes_cd * 100
    print(f"{m:<16} {c:>6.2f} {diff_les:>+9.2f}% {diff_ddes:>+9.2f}%  {s}")

# --- Planar truss validation ---
print("\n\nPlanar Truss Benchmark (2D Validation)")
print("=" * 40)
with open(os.path.join("data", "planar_truss_validation.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"  {row['method']}: Cd = {row['drag_coefficient']}  ({row['source']})")
print("  Difference: -0.52%")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1.2]})

# Colour scheme: reference in grey, this study methods in colour
colors = ["#94a3b8", "#2563eb", "#f97316", "#dc2626"]
edge_colors = ["#64748b", "#1d4ed8", "#ea580c", "#b91c1c"]

# 3D truss comparison
bars = ax1.bar(methods, cd, color=colors, edgecolor=edge_colors, linewidth=1.2, width=0.6)

# Add value labels on bars
for bar, val in zip(bars, cd):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
             f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

# Add percentage annotations relative to DDES
for i, (m, c) in enumerate(zip(methods, cd)):
    if m != "DDES":
        diff = (c - ddes_cd) / ddes_cd * 100
        color = "#16a34a" if abs(diff) < 10 else "#dc2626"
        ax1.text(bars[i].get_x() + bars[i].get_width() / 2, c / 2,
                 f"{diff:+.1f}%\nvs DDES", ha="center", va="center",
                 fontsize=9, color="white", fontweight="bold")

ax1.set_ylabel("Drag Coefficient, $C_D$", fontsize=12)
ax1.set_title("3D Truss — Model Fidelity Comparison", fontsize=13, fontweight="bold")
ax1.set_ylim(0, 3.8)
ax1.grid(True, axis="y", alpha=0.3)

# Planar truss validation (small panel)
planar_methods = ["LES\n(Nakayama)", "DDES\n(This study)"]
planar_cd = [1.93, 1.92]
bars2 = ax2.bar(planar_methods, planar_cd, color=["#94a3b8", "#2563eb"], 
                edgecolor=["#64748b", "#1d4ed8"], linewidth=1.2, width=0.5)
for bar, val in zip(bars2, planar_cd):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax2.set_ylabel("Drag Coefficient, $C_D$", fontsize=12)
ax2.set_title("Planar Truss Validation\n(−0.52% error)", fontsize=12, fontweight="bold")
ax2.set_ylim(0, 2.5)
ax2.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig(os.path.join("figures", "drag_comparison.png"), dpi=200, bbox_inches="tight")
plt.show()
print(f"\nSaved: figures/drag_comparison.png")
