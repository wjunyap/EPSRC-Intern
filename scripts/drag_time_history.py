"""
DDES Drag Coefficient Time History — 3D Truss Structure

Plots the transient drag force from the DDES simulation, showing the
oscillatory behaviour caused by vortex shedding. The mean drag coefficient
is computed from the statistically steady region (t > 100s).

The simulation ran for 130 seconds of physical time at dt = 0.01s,
taking 3 days 17 hours on HPC.

Author: Wei Jun Yap
Project: EPSRC Vacation Internship, University of Liverpool (2024)
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# --- Load data ---
time = []
drag = []
with open(os.path.join("data", "drag_time_history.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        time.append(float(row["time_s"]))
        drag.append(float(row["drag_force_magnitude"]))

t = np.array(time)
F = np.array(drag)

# --- Statistics ---
# Use the settled region for mean calculation (after initial transient)
# The report says mean drag was computed from 100s-120s
mask_settled = (t >= 100) & (t <= 120)
F_settled = F[mask_settled]
mean_F = np.mean(F_settled)
std_F = np.std(F_settled)

print("DDES Drag Force Time History — 3D Truss")
print("=" * 50)
print(f"Total simulation time:  {t[-1]:.0f} s")
print(f"Time step:              {t[1] - t[0]:.4f} s")
print(f"Total data points:      {len(t)}")
print(f"\nStatistics (100s–120s settling window):")
print(f"  Mean drag force:      {mean_F:.4f} N")
print(f"  Std deviation:        {std_F:.4f} N")
print(f"  Fluctuation (σ/μ):    {std_F/mean_F*100:.2f}%")

# --- Estimate Strouhal-like oscillation frequency ---
# Simple zero-crossing analysis on de-meaned signal in settled region
t_s = t[mask_settled]
F_demean = F_settled - mean_F
crossings = np.where(np.diff(np.sign(F_demean)))[0]
if len(crossings) > 2:
    periods = np.diff(t_s[crossings[::2]])  # every other crossing = full period
    if len(periods) > 0:
        avg_period = np.mean(periods)
        freq = 1.0 / avg_period
        print(f"  Est. oscillation freq: {freq:.3f} Hz (period ≈ {avg_period:.2f} s)")

# --- Plot ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]})

# Full time history
ax1 = axes[0]
ax1.plot(t, F, linewidth=0.3, color="#2563eb", alpha=0.8)
ax1.axhline(mean_F, color="#dc2626", linewidth=1.5, linestyle="--", 
            label=f"Mean (100–120s) = {mean_F:.4f} N")
ax1.axvspan(100, 120, alpha=0.1, color="#22c55e", label="Averaging window (100–120s)")
ax1.set_xlabel("Time [s]", fontsize=12)
ax1.set_ylabel("Drag Force Magnitude [N]", fontsize=12)
ax1.set_title("DDES Drag Force Time History — 3D Truss Structure", 
              fontsize=13, fontweight="bold")
ax1.legend(frameon=True, fontsize=10, loc="upper right")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, t[-1])

# Zoomed view of settled region
ax2 = axes[1]
mask_zoom = (t >= 100) & (t <= 120)
ax2.plot(t[mask_zoom], F[mask_zoom], linewidth=0.5, color="#2563eb")
ax2.axhline(mean_F, color="#dc2626", linewidth=1.5, linestyle="--")
ax2.fill_between(t[mask_zoom], mean_F - std_F, mean_F + std_F, 
                 alpha=0.15, color="#2563eb", label=f"±1σ = ±{std_F:.4f} N")
ax2.set_xlabel("Time [s]", fontsize=12)
ax2.set_ylabel("Drag Force [N]", fontsize=12)
ax2.set_title("Zoomed: Averaging Window (100–120s)", fontsize=12, fontweight="bold")
ax2.legend(frameon=True, fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(100, 120)

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig(os.path.join("figures", "drag_time_history.png"), dpi=200, bbox_inches="tight")
plt.show()
print(f"\nSaved: figures/drag_time_history.png")
