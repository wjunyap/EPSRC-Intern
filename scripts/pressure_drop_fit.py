"""
Porous Zone Coefficient Derivation — Pressure Drop Polynomial Fit

Extracts viscous and inertial resistance coefficients for ANSYS Fluent
porous zone modelling by fitting a second-order polynomial to CFD
pressure-drop data across the 3D truss structure.

The pressure drop across a porous medium follows:
    ΔP = A·U² + B·U

where:
    A = 0.5 · ρ · C₂ · Δn    (inertial term)
    B = μ · Δn / α             (viscous term)

Solving for C₂ (inertial resistance) and 1/α (viscous resistance)
gives the inputs required by ANSYS Fluent's porous media model.

Author: Wei Jun Yap
Project: EPSRC Vacation Internship, University of Liverpool (2024)
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# --- Physical constants ---
RHO = 1.225          # Air density [kg/m³]
MU = 1.7894e-5       # Dynamic viscosity [Pa·s]
DELTA_N = 5.92        # Porous zone thickness [m]

# --- Load data ---
velocity = []
delta_p = []
with open(os.path.join("data", "pressure_drop.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        velocity.append(float(row["velocity_measured"]))
        delta_p.append(float(row["delta_pressure"]))

U = np.array(velocity)
dP = np.array(delta_p)

# --- Polynomial fit: ΔP = A·U² + B·U (no constant term) ---
# Rewrite as ΔP = [U², U] · [A, B]ᵀ and solve via least squares
X = np.column_stack([U**2, U])
coeffs, residuals, _, _ = np.linalg.lstsq(X, dP, rcond=None)
A, B = coeffs

# R² calculation
dP_pred = A * U**2 + B * U
ss_res = np.sum((dP - dP_pred) ** 2)
ss_tot = np.sum((dP - np.mean(dP)) ** 2)
r_squared = 1 - ss_res / ss_tot

# --- Derive porous coefficients ---
C2 = (2 * A) / (RHO * DELTA_N)               # Inertial resistance [1/m]
inv_alpha = (B * DELTA_N) / (MU * DELTA_N)    # Simplified: B / MU
# Correct formula: B = μ·Δn/α → 1/α = B·Δn / (μ·Δn²)... 
# Actually: ΔP = (μ/α)·Δn·U + 0.5·ρ·C₂·Δn·U²
# So B = μ·Δn/α → 1/α = B / (μ * Δn) ... but the variable is just B from the fit
# B (from fit) corresponds to μ·Δn/α, so 1/α = B / (μ)  ... no.
# Let me be precise:
# ΔP = (μ/α)·Δn·U + 0.5·ρ·C₂·Δn·U²
# Therefore: B_fit = (μ/α)·Δn  →  1/α = B_fit / (μ · Δn)
inv_alpha = B / (MU * DELTA_N)

print("Porous Zone Coefficient Derivation")
print("=" * 55)
print(f"\nPolynomial fit: ΔP = {A:.4f}·U² + {B:.4f}·U")
print(f"R² = {r_squared:.4f}")
print(f"\nPhysical parameters:")
print(f"  Air density (ρ):        {RHO} kg/m³")
print(f"  Dynamic viscosity (μ):  {MU:.4e} Pa·s")
print(f"  Porous zone thickness:  {DELTA_N} m")
print(f"\nDerived coefficients:")
print(f"  Inertial resistance (C₂):  {C2:.5f} 1/m")
print(f"  Viscous resistance (1/α):  {inv_alpha:.2f} 1/m²")

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 5))

# Data points
ax.scatter(U, dP, s=80, color="#2563eb", zorder=4, label="CFD data", edgecolors="white")

# Fitted curve
U_fine = np.linspace(0, 1.05, 200)
dP_fit = A * U_fine**2 + B * U_fine
ax.plot(U_fine, dP_fit, "-", color="#dc2626", linewidth=2, 
        label=f"Fit: ΔP = {A:.4f}U² + {B:.4f}U  (R² = {r_squared:.4f})")

ax.set_xlabel("Velocity, U [m/s]", fontsize=12)
ax.set_ylabel("Pressure Drop, ΔP [Pa]", fontsize=12)
ax.set_title("Pressure Drop vs Velocity — Porous Zone Coefficient Derivation", 
             fontsize=13, fontweight="bold")
ax.legend(frameon=True, fontsize=10)
ax.grid(True, alpha=0.3)

# Annotate derived coefficients
textstr = (f"Inertial resistance C₂ = {C2:.5f} m⁻¹\n"
           f"Viscous resistance 1/α = {inv_alpha:.2f} m⁻²")
ax.text(0.05, 0.85, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4", 
        facecolor="lightyellow", edgecolor="gray", alpha=0.9))

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig(os.path.join("figures", "pressure_drop_fit.png"), dpi=200, bbox_inches="tight")
plt.show()
print(f"\nSaved: figures/pressure_drop_fit.png")
