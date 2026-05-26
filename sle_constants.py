"""
Shared constants and κ* derivation for all SLE scripts.
"""
import math, numpy as np

sqrt3 = math.sqrt(3)
LAM   = (1 + sqrt3 + math.sqrt(2 + 2*sqrt3)) / 2
LAM2  = LAM**2
H_TOP = math.log(LAM)
DELTA_CHI = 1
PI_CIRC   = DELTA_CHI / LAM2
G_E       = -504

# Derivation of κ* from the Eisenstein coupling gE = -504
# --------------------------------------------------------
# gE = -504 is the first Fourier coefficient of E_6(τ), the weight-6
# Eisenstein series.  The ratio |gE| / c_{E_4} = 504/240 = 2.1 measures
# the relative coupling between the weight-6 and weight-4 modular operators
# at the Hat→Spectre phase boundary.
#
# In 2D CFT, the boundary-changing operator between the two phases has
# conformal dimension h related to κ by:
#     h = (6 - κ) / (2κ)          [standard SLE boundary weight]
#
# We identify h with the Eisenstein-derived coupling:
#     h = 1 / (1 + |gE|/c_{E_4}) = 1 / (1 + 504/240) = 240/744
#
# Solving for κ:
#     κ = 6 / (2h + 1)

E4_coeff  = 240              # leading coefficient of E_4(τ)
gE_ratio  = abs(G_E) / E4_coeff   # = 504/240 = 2.1
h_boundary = 1 / (1 + gE_ratio)   # ≈ 0.3226
kappa_star = 6 / (2*h_boundary + 1)   # ≈ 3.647
c_star     = (6 - kappa_star)*(3*kappa_star - 8) / (2*kappa_star)  # ≈ 0.949
D_f_star   = 1 + kappa_star / 8   # ≈ 1.456

if __name__ == "__main__":
    print(f"λ         = {LAM:.8f}")
    print(f"h_top     = {H_TOP:.8f} nats")
    print(f"|gE|/c_E4 = {gE_ratio:.6f}")
    print(f"h_bound   = {h_boundary:.6f}")
    print(f"κ*        = {kappa_star:.6f}")
    print(f"c*        = {c_star:.6f}")
    print(f"D_f*      = {D_f_star:.6f}")
    print(f"Π_circ    = {PI_CIRC:.6f} ({PI_CIRC*100:.2f}%)")
