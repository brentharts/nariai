"""
gap_label_spectrum.py
=====================
Computes the gap-label spectrum of the Spectre tiling's C*-algebra,
plots it against a periodic (rational) baseline, and exports a
comparison table for use in the paper.

The gap-labeling theorem (Bellissard, van Elst, Schulz-Baldes 1994)
states that the integrated density of states (IDS) at each spectral
gap of the Hamiltonian on ℓ²(Ω) is an element of the group

    𝒢 = ℤ[λ⁻¹] ⊂ ℝ

where λ is the Spectre inflation factor.  Each gap is labelled by a
unique element   g = Σₖ nₖ λ⁻ᵏ   (nₖ ∈ ℤ).

Usage:
    python3 gap_label_spectrum.py          # text output only
    python3 gap_label_spectrum.py --plot   # also show matplotlib figure
"""

import math
import sys
import numpy as np
from itertools import product

# ── Spectre inflation factor ──────────────────────────────────
# REBUILT on nariai_constants.  Two corrections, not one:
#
#   (a) the inflation factor was the retracted one;
#   (b) the base of the gap-label module is the AREA factor lambda_A, not
#       the linear one.  Gap labels lie in Z[1/lambda_A] (Bellissard-van
#       Elst-Schulz-Baldes), so sum(n_k lam**-k) must be summed over powers
#       of 4+sqrt(15).  Substituting the linear factor here would still be
#       wrong even with the retraction fixed, which is why this file is not
#       a one-line change.
from nariai_constants import GAP_BASE as LAM, GROWTH_RATE

# log(lambda_A) is the growth rate of the tile count, NOT a topological
# entropy: primitive substitution tilings are uniquely ergodic and their
# translation action has zero topological entropy.
H_TOP = GROWTH_RATE


def gap_label(coefficients):
    """Evaluate g = Σ nₖ λ⁻ᵏ for a list of integer coefficients."""
    return sum(c * LAM**(-k) for k, c in enumerate(coefficients))


def generate_gap_labels(max_order=6, coeff_range=(-1, 0, 1)):
    """
    Generate all gap labels up to `max_order` in the basis {λ⁻ᵏ}.
    Returns sorted unique values in [0, 1].
    """
    labels = set()
    for order in range(1, max_order + 1):
        for coeffs in product(coeff_range, repeat=order):
            g = gap_label(coeffs)
            if 0 < g < 1:
                labels.add(round(g, 12))
    return sorted(labels)


def periodic_labels(n_harmonics=30):
    """Rational labels p/q with q ≤ n_harmonics — the periodic baseline."""
    labels = set()
    for q in range(1, n_harmonics + 1):
        for p in range(1, q):
            labels.add(p / q)
    return sorted(labels)


# ── Generate labels ───────────────────────────────────────────
print("=" * 65)
print("  Gap-Label Spectrum  𝒢 = ℤ[λ⁻¹]  for the Spectre Tiling")
print("=" * 65)
print(f"\n  λ = {LAM:.10f}")
print(f"  Growth rate log(λ_A) = {H_TOP:.8f} nats  (not an entropy)\n")

# Primary hierarchy: λ⁻ᵏ for k = 0..7
print("  Primary gap labels  λ⁻ᵏ:")
print(f"  {'k':>3}  {'λ⁻ᵏ':>14}  {'Ratio λ⁻(k-1)/λ⁻k':>20}")
primary = [LAM**(-k) for k in range(8)]
for k, g in enumerate(primary):
    ratio = primary[k-1] / g if k > 0 else "—"
    ratio_str = f"{ratio:.8f}" if k > 0 else "—"
    print(f"  {k:>3}  {g:>14.10f}  {ratio_str:>20}")

# Extended labels from integer combinations
print("\n  Extended ℤ[λ⁻¹] labels (coefficients in {-1,0,1}, order ≤ 4):")
ext = generate_gap_labels(max_order=4)
print(f"  Total unique labels in (0,1): {len(ext)}")
print(f"  First 12: {[f'{g:.6f}' for g in ext[:12]]}")

# Density comparison: aperiodic vs periodic
ap_density = len(ext) / 1.0       # labels per unit interval
per_labels  = periodic_labels(20)
print(f"\n  Aperiodic labels (|coeff|≤1, k≤4):  {len(ext)} in (0,1)")
print(f"  Periodic labels (q≤20):             {len(per_labels)} in (0,1)")
print("  Both counts are artifacts of where the sums were truncated, and")
print("  neither is a property of the spectrum: ℤ[1/λ_A] = ℤ[√15] is DENSE")
print("  in ℝ, so relaxing either cutoff grows its count without bound.")
print("  The real contrast is structural -- ℤ[√15] against ℤ[1/q] -- and")
print("  is not captured by comparing two finite tallies.")

# Distinguishing feature: check irrationality of all primary labels
# Irrationality check, on a RELATIVE error.  The previous version compared
# |g − p/q| against a fixed absolute tolerance, which is not scale free: as
# λ⁻ᵏ shrinks the nearest rational with small q becomes 0/1 and the absolute
# error becomes λ⁻ᵏ itself, so the test declared the label periodic for
# k ≥ 14.  See argument_audit.py, finding 'gap-irrationality'.
print("\n  Irrationality check (relative distance to the nearest p/q, q<1000):")
for k in range(1, 8):
    g = LAM**(-k)
    best_q = min(range(1, 1000), key=lambda q: abs(g - round(g * q) / q))
    best_approx = round(g * best_q) / best_q
    rel = abs(g - best_approx) / g
    print(f"  λ⁻{k} = {g:.10f}  best ≈ {round(g*best_q)}/{best_q}"
          f"  Δ/g = {rel:.2e}")

print(f"\n  All primary labels are irrational: λ⁻ᵏ = (4−√15)ᵏ = m + n√15 with")
print(f"  n ≠ 0 for every k ≥ 1, so no power is rational.  The table above is")
print(f"  an illustration of that fact, not a proof of it -- a finite search")
print(f"  over q can never establish irrationality.")

# ── Optional plot ─────────────────────────────────────────────
if "--plot" in sys.argv:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(
            r"Gap-Label Spectrum $\mathcal{G} = \mathbb{Z}[\lambda^{-1}]$"
            "\nvs Rational (Periodic) Baseline",
            fontsize=13
        )

        # Axis 0: primary hierarchy λ⁻ᵏ
        ax = axes[0]
        ax.set_title(r"Primary hierarchy $\lambda^{-k}$, $k=0,\ldots,7$", fontsize=10)
        for k, g in enumerate(primary):
            ax.axvline(g, color="steelblue", lw=2, alpha=0.85,
                       label=f"λ⁻{k}" if k < 4 else None)
            ax.text(g, 0.6, f"k={k}", ha="center", va="bottom",
                    fontsize=7, color="steelblue", rotation=90)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel("Aperiodic", fontsize=9)

        # Axis 1: extended ℤ[λ⁻¹] labels
        ax = axes[1]
        ax.set_title(r"All $\mathbb{Z}[\lambda^{-1}]$ labels, $|n_k|\leq 1$, order $\leq 4$", fontsize=10)
        for g in ext:
            ax.axvline(g, color="darkorange", lw=0.7, alpha=0.6)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel("Aperiodic", fontsize=9)

        # Axis 2: periodic rational baseline
        ax = axes[2]
        ax.set_title("Periodic baseline: rational labels $p/q$, $q \\leq 20$", fontsize=10)
        for g in per_labels:
            ax.axvline(g, color="firebrick", lw=0.5, alpha=0.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel("Periodic", fontsize=9)
        ax.set_xlabel("Integrated density of states (fraction of band)", fontsize=10)
        ax.set_xlim(0, 1)

        blue_patch   = mpatches.Patch(color="steelblue",   label="Aperiodic primary λ⁻ᵏ")
        orange_patch = mpatches.Patch(color="darkorange",  label="Aperiodic extended ℤ[λ⁻¹]")
        red_patch    = mpatches.Patch(color="firebrick",   label="Periodic rational")
        fig.legend(handles=[blue_patch, orange_patch, red_patch],
                   loc="lower center", ncol=3, fontsize=9, framealpha=0.9)

        plt.tight_layout(rect=[0, 0.06, 1, 1])
        plt.savefig("gap_label_spectrum.pdf", dpi=150, bbox_inches="tight")
        print("\n  Figure saved to gap_label_spectrum.pdf")
        plt.show()

    except ImportError:
        print("\n  matplotlib not installed — skipping plot.")
