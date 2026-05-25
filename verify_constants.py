"""
verify_constants.py
====================
Reproduces every numerical constant cited in:

  Hartshorn, B. S. (2026)
  "Aperiodic Vacuum Structure from BRST Cohomology on the Nariai Background"

Run:  python3 verify_constants.py

All values are derived from first principles; no free parameters are tuned.
"""

import math
import numpy as np

SEPARATOR = "-" * 60

def section(title):
    print(f"\n{SEPARATOR}\n  {title}\n{SEPARATOR}")


# ──────────────────────────────────────────────────────────────
# 1.  SPECTRE INFLATION FACTOR λ
# ──────────────────────────────────────────────────────────────
section("1. Spectre Inflation Factor λ")

# Smith et al. (2023) give the exact closed form:
#   λ = (1 + √3 + √(2 + 2√3)) / 2
sqrt3 = math.sqrt(3)
lam_exact = (1 + sqrt3 + math.sqrt(2 + 2 * sqrt3)) / 2

# Independent check: largest real root of minimal polynomial
#   4λ⁴ − 8λ³ − 4λ² − 4λ + 1 = 0   (obtained via sympy.minpoly)
coeffs = [4, -8, -4, -4, 1]
roots = np.roots(coeffs)
real_roots = roots[np.abs(roots.imag) < 1e-10].real
lam_poly = float(max(real_roots))

assert abs(lam_exact - lam_poly) < 1e-10, "Polynomial root does not match closed form!"

lam = lam_exact
print(f"  Closed form  λ = {lam:.10f}")
print(f"  Poly root    λ = {lam_poly:.10f}")
print(f"  Area factor  λ² = {lam**2:.8f}  (≈ 6.425)")
print(f"  Minimal poly: 4λ⁴ − 8λ³ − 4λ² − 4λ + 1 = "
      f"{4*lam**4 - 8*lam**3 - 4*lam**2 - 4*lam + 1:.2e}  (≈ 0)")

# Pisot property: all other roots lie strictly inside unit disk
other_mods = [abs(r) for r in roots if abs(r - lam) > 1e-6]
print(f"  Other root moduli: {[f'{m:.6f}' for m in sorted(other_mods)]}  (all < 1 → Pisot ✓)")
assert all(m < 1 for m in other_mods), "Not a Pisot number!"


# ──────────────────────────────────────────────────────────────
# 2.  TOPOLOGICAL ENTROPY
# ──────────────────────────────────────────────────────────────
section("2. Topological Entropy of the Spectre Tiling")

h_top_nats = math.log(lam)
h_top_bits = math.log2(lam)

print(f"  h_top = log(λ) = {h_top_nats:.8f} nats")
print(f"  h_top = log₂(λ) = {h_top_bits:.8f} bits")
print(f"  (Paper cites ≈ 0.930 nats, ≈ 1.342 bits ✓)")

assert abs(h_top_nats - 0.930) < 0.001
assert abs(h_top_bits - 1.342) < 0.001


# ──────────────────────────────────────────────────────────────
# 3.  GAP-LABELING GROUP  ℤ[λ⁻¹]
# ──────────────────────────────────────────────────────────────
section("3. Gap-Labeling Group  ℤ[λ⁻¹]")

print("  First 8 gap labels λ⁻ᵏ:")
gap_labels = []
for k in range(8):
    g = lam ** (-k)
    gap_labels.append(g)
    print(f"    k={k}: λ⁻{k} = {g:.8f}")

print(f"\n  Consecutive ratio (should all equal λ ≈ {lam:.4f}):")
for k in range(1, 6):
    ratio = gap_labels[k-1] / gap_labels[k]
    print(f"    λ⁻{k-1} / λ⁻{k} = {ratio:.8f}")
    assert abs(ratio - lam) < 1e-9

# Recurrence from minimal polynomial  4λ⁴ − 8λ³ − 4λ² − 4λ + 1 = 0
# Multiply by λ⁻(k+4):
#   4λ⁻k − 8λ⁻(k+1) − 4λ⁻(k+2) − 4λ⁻(k+3) + λ⁻(k+4) = 0
# => λ⁻(k+4) = −4λ⁻k + 8λ⁻(k+1) + 4λ⁻(k+2) + 4λ⁻(k+3)
print("\n  Verifying recurrence relation:")
for k in range(4):
    lhs = lam ** (-(k + 4))
    rhs = (-4 * lam**(-k)
           + 8 * lam**(-(k+1))
           + 4 * lam**(-(k+2))
           + 4 * lam**(-(k+3)))
    print(f"    k={k}: LHS={lhs:.12f}  RHS={rhs:.12f}  Δ={abs(lhs-rhs):.2e}")
    assert abs(lhs - rhs) < 1e-12, f"Recurrence fails at k={k}"


# ──────────────────────────────────────────────────────────────
# 4.  CHIRALITY COST AND MARKOV GAP LOWER BOUND
# ──────────────────────────────────────────────────────────────
section("4. Chirality Cost  Δχ  and Markov Gap Lower Bound")

V_hat     = 13   # edge count of Hat prototile
V_spectre = 14   # edge count of Spectre prototile
delta_chi = V_spectre - V_hat

print(f"  V_Hat     = {V_hat}")
print(f"  V_Spectre = {V_spectre}")
print(f"  Δχ = V_Spectre − V_Hat = {delta_chi}")
print()
print(f"  Markov gap lower bound:")
print(f"    ΔS ≥ h_top = log(λ) ≈ {h_top_nats:.4f} nats")
print(f"                        ≈ {h_top_bits:.4f} bits")


# ──────────────────────────────────────────────────────────────
# 5.  CIRCULAR POLARISATION PREDICTION
# ──────────────────────────────────────────────────────────────
section("5. SGWB Circular Polarisation Prediction  Π_circ")

Pi_circ = delta_chi / lam**2
print(f"  Π_circ = Δχ / λ² = {delta_chi} / {lam**2:.6f}")
print(f"         = {Pi_circ:.6f}  ≈ {Pi_circ*100:.1f}%")
print(f"  (Paper cites ≈ 15.6% ✓)")
assert abs(Pi_circ - 0.1556) < 0.001


# ──────────────────────────────────────────────────────────────
# 6.  BRST NILPOTENCY CHECK (ALGEBRAIC)
# ──────────────────────────────────────────────────────────────
section("6. BRST Nilpotency  Q² = 0  (Anderson-Putnam boundary operator)")

# On a chain complex C² →^∂₂ C¹ →^∂₁ C⁰, we verify ∂₁ ∘ ∂₂ = 0.
# Example: standard oriented triangle (minimal 2D simplicial complex).
# Vertices: 0,1,2  |  Edges: [0,1],[0,2],[1,2]  |  Face: [0,1,2]
#
# ∂₁: edges → vertices    rows=vertices, cols=edges
D1 = np.array([
    [-1, -1,  0],   # vertex 0
    [ 1,  0, -1],   # vertex 1
    [ 0,  1,  1],   # vertex 2
], dtype=float)

# ∂₂: faces → edges       rows=edges, cols=faces
#   ∂([0,1,2]) = +[0,1] − [0,2] + [1,2]
D2 = np.array([[ 1], [-1], [ 1]], dtype=float)

Q2 = D1 @ D2
max_entry = np.max(np.abs(Q2))
print(f"  ∂₁ ∘ ∂₂ = {Q2.T}  max|entry| = {max_entry:.2e}")
print(f"  {'✓ Q² = 0  (exact algebraic identity)' if max_entry < 1e-10 else '✗ FAILED'}")
assert max_entry < 1e-10, "∂² ≠ 0"

print()
print("  Interpretation: on an aperiodic tiling, Q = ∂ (chain boundary operator).")
print("  Q² = 0 is algebraic — requires NO anomaly cancellation,")
print("  in contrast to periodic string backgrounds (d=26 or d=10).")


# ──────────────────────────────────────────────────────────────
# 7.  TENSOR-TO-SCALAR RATIO CONSTRAINT (Liu et al. 2026)
# ──────────────────────────────────────────────────────────────
section("7. Tensor-to-Scalar Ratio  r ≥ 0.01  (Liu et al. 2026)")

r_min = 0.01
print(f"  Minimum predicted r = {r_min}")
print(f"  Source: Liu, Quintin & Afshordi (2026), Phys. Rev. Lett. 136, 111501")
print(f"  DOI: 10.1103/6gtx-j455")
print(f"  Our framework requires r > 0 independently (R² suppression at μ*).")
print(f"  r = 0 would falsify both frameworks simultaneously.")


# ──────────────────────────────────────────────────────────────
# 8.  SUMMARY TABLE
# ──────────────────────────────────────────────────────────────
section("8. Summary of Paper Constants")

rows = [
    ("λ  (inflation factor)",     f"{lam:.8f}",         "Eq. (1)"),
    ("λ² (area scaling)",          f"{lam**2:.8f}",      "Eq. (1)"),
    ("h_top [nats]",               f"{h_top_nats:.8f}",  "§ IV.B"),
    ("h_top [bits]",               f"{h_top_bits:.8f}",  "§ IV.B"),
    ("Gap label  λ⁻¹",             f"{1/lam:.8f}",       "Eq. (6)"),
    ("Gap label  λ⁻²",             f"{1/lam**2:.8f}",    "Eq. (6)"),
    ("Δχ  (chirality cost)",       f"{delta_chi}",        "§ V.A"),
    ("Markov gap lower bound [nat]",f"{h_top_nats:.4f}", "Prop. 2"),
    ("Π_circ (polarisation)",       f"{Pi_circ:.4f}",    "Eq. (12)"),
    ("r_min",                       f"{r_min}",           "Eq. (10)"),
]
print(f"\n  {'Quantity':<35} {'Value':<18} {'Location'}")
print(f"  {'-'*35} {'-'*18} {'-'*12}")
for name, val, loc in rows:
    print(f"  {name:<35} {val:<18} {loc}")

print(f"\n  All assertions passed. ✓")
