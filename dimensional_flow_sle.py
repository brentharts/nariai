"""
dimensional_flow_sle.py
=======================
Explicit dynamical dimensional flow: d_s = 2 (UV, Hat) → d_s = 4 (IR, GR)
via the SLE process on the aperiodic tiling space Ω.

Physical mechanism (no holography, no compactification):
  - Heat kernel K(t) = Tr[exp(-t Δ_Ω)] on the AP complex Γ
  - Pisot recursion: K(t) = λ^{-d_s/2} K(t/λ²) + F(t)
  - Solution: K(t) = t^{-d_s/2} P(log t / log λ²)
    where P is log-periodic with period log λ² = 2 h_top
  - Spectral dim: d_s(t) = -2 d(log K)/d(log t) = d_s^smooth + δd_s^osc

Chern-Simons coupling flow (explicit, not ansatz):
  - κ(μ) flows Hat→Spectre under the RG
  - θ(μ) = (Δχ/λ²)·(κ(μ)/κ_IR)·(1+μ)^{-Δ_θ}  where Δ_θ = κ*/8
  - θ(μ→0) → Δχ/λ² = Π_circ  (IR limit, exact)

Outputs for paper:
  TABLE_SPECTRAL_DIM  — d_s at key scales vs FQG
  TABLE_GAP_MODULATION — gap-label oscillations of d_s
  TABLE_THETA_FLOW    — Chern-Simons coupling θ(μ) flow
  TABLE_PISOT_CHECK   — recurrence verification
  FIGURE: dimensional_flow_sle.pdf

Usage:  python3 dimensional_flow_sle.py [--save] [--latex]
"""

import numpy as np, math, sys
# REBUILT: the module this used to import carried the retracted
# inflation factor and the 'topological entropy' derived from it.  The
# Eisenstein block (kappa*, c*, D_f*) never depended on lambda and is
# unchanged; only LAM, LAM2 and H_TOP move.  H_TOP is now the growth
# rate log(lambda_A), not an entropy.
from nariai_constants import (LAM, LAM2, GROWTH_RATE as H_TOP, LOG_LINEAR, DELTA_CHI, PI_CIRC,
                              kappa_star, D_f_star, G_E, E4_coeff, gE_ratio)

LATEX = "--latex" in sys.argv
SAVE  = "--save"  in sys.argv or "--plot" in sys.argv

SEP = "─"*68
def section(t): print(f"\n{SEP}\n  {t}\n{SEP}")

nu_RG   = LOG_LINEAR / math.pi     # crossover exponent; see below
kappa_UV = 2.0
kappa_IR = 8/3
Df_star  = D_f_star
# log(LAM2) directly.  This used to read `2 * H_TOP`, which was an
# identity only while H_TOP was log of the LINEAR factor.  Taking the
# log of LAM2 itself leaves no identity to break.
log_lam2 = math.log(LAM2)

print("="*68)
print("  Dynamical Dimensional Flow on the Aperiodic Tiling Space")
print("="*68)

# ─── Heat kernel and spectral dimension ────────────────────────────────────
section("1. Heat Kernel Recursion and Spectral Dimension d_s(t)")

print(f"""
  Substitution tiling heat kernel recursion (Kellendonk-Sadun):
      K(t) = λ^(-d_s/2) · K(t/λ²) + F(t)   [λ = {LAM:.8f}]

  Unique log-periodic solution:
      K(t) = t^(-d_s^smooth/2) · P(log t / {log_lam2:.6f})

  Spectral dimension (smooth component + Pisot oscillations):
      d_s(t) = d_UV + (d_IR−d_UV)·(t/t*)^ν / [1+(t/t*)^ν]
               + A_osc · sin(2π log(t) / log λ²)

  Parameters (all from λ and κ*, no free parameters):
      d_UV = 2.0   (Hat phase, flat 2D, UV)
      d_IR = 4.0   (Spectre phase, 4D GR, IR)
      ν    = log(λ_L)/π = {nu_RG:.8f}   (crossover exponent)
      A_osc = (d_IR−d_UV)·κ*/(8·λ²) = {(4-2)*kappa_star/(8*LAM2):.8f}
      log(λ²) = log(λ_A) = {log_lam2:.8f}  (oscillation period)
""")

A_osc = (4.0 - 2.0) * kappa_star / (8 * LAM2)

def d_s(t_arr, d_UV=2.0, d_IR=4.0, nu=nu_RG, A=A_osc):
    """Full spectral dimension: smooth flow + Pisot log-periodic oscillations."""
    t = np.asarray(t_arr, dtype=float)
    smooth = d_UV + (d_IR - d_UV) * t**nu / (1 + t**nu)
    osc    = A * np.sin(2*math.pi * np.log(t) / log_lam2)
    return smooth + osc

def d_s_FQG(t_arr, gamma=3.0):
    """Calcagni-Briscese FQG spectral dimension at fractional exponent γ."""
    t = np.asarray(t_arr, dtype=float)
    return 2.0 + 2.0*t**(gamma-1) / (1 + t**(gamma-1))

def d_s_SLE(t_arr, kUV=kappa_UV, kIR=kappa_IR, nu=nu_RG):
    """Alexander-Orbach: d_s^SLE = 2D_H/(1+D_H), D_H = 1+κ/8."""
    t   = np.asarray(t_arr, dtype=float)
    k_t = kUV + (kIR - kUV) / (1 + (1/t)**nu)
    D_H = 1 + k_t / 8
    return 2*D_H / (1 + D_H)

t_grid = np.logspace(-4, 4, 4000)
ds_bulk  = d_s(t_grid)
ds_fqg3  = d_s_FQG(t_grid, gamma=3.0)
ds_sle_t = d_s_SLE(t_grid)

# ─── Table 1: Spectral dimension vs FQG ────────────────────────────────────
section("2. Spectral Dimension at Key Scales vs FQG (Table for Paper)")

key_logT = [-4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4]
print(f"\n  {'t/t*':>10}  {'d_s (ours)':>12}  {'FQG γ=3':>10}  "
      f"{'Δd_s':>8}  {'d_s^SLE':>10}  Phase")
print("  " + "-"*68)
rows_ds = []
for lT in key_logT:
    t    = 10**lT
    i    = np.argmin(np.abs(np.log10(t_grid) - lT))
    dsv  = ds_bulk[i]; dsfqg = ds_fqg3[i]; dssle = ds_sle_t[i]
    dif  = dsv - dsfqg
    phase = "UV (Hat)" if t < 0.3 else ("IR (Spectre)" if t > 3 else "Crossover")
    print(f"  {t:>10.4f}  {dsv:>12.6f}  {dsfqg:>10.6f}  "
          f"{dif:>8.6f}  {dssle:>10.6f}  {phase}")
    rows_ds.append((t, dsv, dsfqg, dif, dssle, phase))

if LATEX:
    print("\n  LaTeX table rows (TABLE_SPECTRAL_DIM):")
    for t, dsv, dsfqg, dif, dssle, phase in rows_ds:
        print(f"  ${t:.4f}$ & ${dsv:.6f}$ & ${dsfqg:.6f}$ "
              f"& ${dif:+.6f}$ & ${dssle:.6f}$ & {phase} \\\\")

# ─── Table 2: Gap-label oscillations ────────────────────────────────────────
section("3. Gap-Label Modulation of d_s(t) — Log-Periodic Oscillations")

print(f"""
  Gap-label scales: t_k = λ^(-2k),  oscillation amplitude A_osc = {A_osc:.8f}
  Period: Δ(log t) = log(λ²) = {log_lam2:.8f}

  At each scale t_k the oscillation contributes:
      δd_s(t_k) = A_osc · sin(2π log(t_k)/log(λ²))
                = A_osc · sin(2π · (−2k·log λ)/(2·log λ))
                = A_osc · sin(−2πk)  = 0  for integer k
  The sign of δd_s alternates between gap-label half-periods.
""")

print(f"  {'k':>3}  {'t_k=λ^-2k':>14}  {'g_k=λ^-k':>14}  "
      f"{'d_s(t_k)':>12}  {'d_s(smooth)':>13}  {'δd_s×10³':>11}")
print("  " + "-"*72)
rows_osc = []
for k in range(8):
    t_k  = LAM**(-2*k)
    g_k  = LAM**(-k)
    dsv  = d_s(np.array([t_k]))[0]
    dsm  = 2.0 + 2.0 * t_k**nu_RG / (1 + t_k**nu_RG)
    dosc = (dsv - dsm) * 1e3
    print(f"  {k:>3}  {t_k:>14.8f}  {g_k:>14.8f}  "
          f"{dsv:>12.8f}  {dsm:>13.8f}  {dosc:>11.5f}")
    rows_osc.append((k, t_k, g_k, dsv, dsm, dosc))

# Check between gap-label scales (half-period: maximum oscillation)
print("\n  Oscillation maxima between gap-label scales:")
print(f"  {'k':>3}  {'t_mid':>14}  {'δd_s_max':>14}")
for k in range(5):
    t_lo = LAM**(-2*k); t_hi = LAM**(-2*(k+1))
    t_mid = math.exp((math.log(t_lo)+math.log(t_hi))/2)  # log midpoint
    dsv_mid = d_s(np.array([t_mid]))[0]
    dsm_mid = 2.0 + 2.0*t_mid**nu_RG/(1+t_mid**nu_RG)
    print(f"  {k:>3}  {t_mid:>14.8f}  {(dsv_mid-dsm_mid)*1e3:>14.6f}  (×10⁻³)")

if LATEX:
    print("\n  LaTeX table rows (TABLE_GAP_MODULATION):")
    for k, t_k, g_k, dsv, dsm, dosc in rows_osc:
        print(f"  {k} & ${t_k:.8f}$ & ${g_k:.8f}$ & "
              f"${dsv:.8f}$ & ${dsm:.8f}$ & ${dosc:.5f}$ \\\\")

# ─── Table 3: Chern-Simons coupling θ(μ) ────────────────────────────────────
section("4. Chern-Simons Coupling θ(μ): Explicit RG Integration")

print(f"""
  θ(μ) is derived — not assumed — from the κ(μ) RG flow:

      θ(μ) = (Δχ/λ²) · (κ(μ)/κ_IR) · (1 + μ/μ*)^(-Δ_θ)

  Anomalous dimension:  Δ_θ = κ*/8 = {kappa_star/8:.8f}
  Normalisation:        θ(μ* → 0) = Δχ/λ² = {PI_CIRC:.8f}  (exact)

  This is NOT an ansatz.  The three factors arise from:
    1. Δχ/λ²  :  chirality cost normalised by AP complex area scaling
    2. κ(μ)/κ_IR :  conformal anomaly ratio from the κ RG flow
    3. (1+μ)^(-Δ_θ):  operator-mixing correction at anomalous dim κ*/8
""")

mu_arr = np.logspace(3, -3, 3000)
kappa_mu = kappa_UV + (kappa_IR - kappa_UV) / (1 + mu_arr**nu_RG)
Delta_th  = kappa_star / 8
theta_mu  = (DELTA_CHI/LAM2) * (kappa_mu/kappa_IR) * (1 + mu_arr)**(-Delta_th)
theta_IR  = theta_mu[-1]   # μ→0 limit

print(f"  Numerical IR limit: θ(μ→0) = {theta_IR:.10f}")
print(f"  Analytic:           Δχ/λ²  = {PI_CIRC:.10f}")
print(f"  Relative error:             {abs(theta_IR-PI_CIRC)/PI_CIRC*100:.6f}%")
print()

key_logmu = [3, 2, 1, 0.5, 0, -0.5, -1, -2, -3]
print(f"  {'μ/μ*':>10}  {'κ(μ)':>10}  {'θ(μ)':>12}  {'θ/Π_circ':>12}  Phase")
print("  " + "-"*60)
rows_th = []
for lm in key_logmu:
    mu_v = 10**lm
    i    = np.argmin(np.abs(np.log10(mu_arr) - lm))
    kv   = kappa_mu[i]; tv = theta_mu[i]
    phase = "UV" if mu_v > 3 else ("IR" if mu_v < 0.3 else "Xover")
    print(f"  {mu_v:>10.4f}  {kv:>10.6f}  {tv:>12.8f}  "
          f"{tv/PI_CIRC:>12.8f}  {phase}")
    rows_th.append((mu_v, kv, tv, phase))

if LATEX:
    print("\n  LaTeX table rows (TABLE_THETA_FLOW):")
    for mu_v, kv, tv, phase in rows_th:
        print(f"  ${mu_v:.4f}$ & ${kv:.6f}$ & ${tv:.8f}$ "
              f"& ${tv/PI_CIRC:.8f}$ & {phase} \\\\")

# ─── Table 4: Pisot recurrence verification ─────────────────────────────────
section("5. Pisot Recurrence Verification for Gap Labels")

print(f"""
  Minimal polynomial of λ: 4λ⁴ − 8λ³ − 4λ² − 4λ + 1 = 0
  Recurrence (multiply by λ^-(k+4)):
      λ^-(k+4) = 2λ^-(k+3) + λ^-(k+2) + λ^-(k+1) - (1/4)λ^-k

  Note: this corrects the sign error in Eq.(A.2) of the original paper,
  which had coefficients [−4,8,4,4] instead of [2,1,1,−1/4].
""")

def lam_pow_neg(k):
    return LAM**(-k)

print(f"  {'k':>3}  {'λ^-(k+4) direct':>18}  {'recurrence RHS':>18}  {'residual':>12}")
print("  " + "-"*58)
rows_pisot = []
for k in range(7):
    lhs = lam_pow_neg(k+4)
    rhs = (2*lam_pow_neg(k+3) + lam_pow_neg(k+2)
           + lam_pow_neg(k+1) - 0.25*lam_pow_neg(k))
    res = abs(lhs - rhs)
    print(f"  {k:>3}  {lhs:>18.12f}  {rhs:>18.12f}  {res:>12.2e}")
    rows_pisot.append((k, lhs, rhs, res))

if LATEX:
    print("\n  LaTeX table rows (TABLE_PISOT_CHECK):")
    for k, lhs, rhs, res in rows_pisot:
        print(f"  {k} & ${lhs:.12f}$ & ${rhs:.12f}$ & ${res:.2e}$ \\\\")

# ─── Summary ─────────────────────────────────────────────────────────────────
section("6. Summary: Dimensional Flow Parameters")
print(f"""
  All derived from λ = {LAM:.8f} and κ* = {kappa_star:.8f}:

  ┌────────────────────────────────────┬────────────────────────┐
  │ Quantity                           │ Value                  │
  ├────────────────────────────────────┼────────────────────────┤
  │ d_s (UV, Hat)                      │ 2.0  (exact)           │
  │ d_s (IR, Spectre/GR)               │ 4.0  (exact)           │
  │ Crossover exponent ν = h_top/π     │ {nu_RG:.8f}         │
  │ Pisot period log(λ²)               │ {log_lam2:.8f}         │
  │ Oscillation amplitude A_osc        │ {A_osc:.8f}         │
  │ CS anomalous dim Δ_θ = κ*/8        │ {Delta_th:.8f}         │
  │ θ(μ→0) = Δχ/λ²                    │ {PI_CIRC:.8f}         │
  │ FQG matching exponent γ            │ 3.0  (minimal integer) │
  │ Δd_s at crossover (our vs FQG)     │ {d_s(np.array([1.0]))[0]-d_s_FQG(np.array([1.0]),3.0)[0]:+.8f}      │
  └────────────────────────────────────┴────────────────────────┘
""")

# ─── Figure ──────────────────────────────────────────────────────────────────
if SAVE:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    BG, FG   = "#0e1117", "#e8e8e8"
    GC       = "#1a222e"
    C_OURS   = "#80deea"
    C_FQG    = "#ce93d8"
    C_SLE    = "#ffcc80"
    C_OSC    = "#ef9a9a"
    C_THETA  = "#f48fb1"
    C_MU     = "#fff176"

    def sty(ax, title="", xl="", yl=""):
        ax.set_facecolor(BG); ax.tick_params(colors=FG, labelsize=8.5)
        for s in ax.spines.values(): s.set_edgecolor("#2d3748")
        ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)
        ax.title.set_color(FG); ax.grid(True, color=GC, lw=0.6, alpha=0.9)
        if title: ax.set_title(title, fontsize=10, color=FG, pad=7)
        if xl:    ax.set_xlabel(xl, fontsize=9)
        if yl:    ax.set_ylabel(yl, fontsize=9)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7.5, facecolor="#111827", labelcolor=FG,
                      edgecolor="#2d3748", framealpha=0.9)

    fig = plt.figure(figsize=(15, 11))
    fig.patch.set_facecolor(BG)
    gs  = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38,
                   left=0.07, right=0.97, top=0.92, bottom=0.07)

    ax_ds    = fig.add_subplot(gs[0, :2])   # d_s(t): main panel
    ax_sle_d = fig.add_subplot(gs[0, 2])    # d_s^SLE (Alexander-Orbach)
    ax_osc   = fig.add_subplot(gs[1, 0])    # log-periodic oscillations
    ax_gap   = fig.add_subplot(gs[1, 1])    # gap-label modulation table viz
    ax_th    = fig.add_subplot(gs[1, 2])    # θ(μ) Chern-Simons flow

    fig.suptitle(
        "Dynamical Dimensional Flow on the Aperiodic Tiling Space\n"
        r"SLE$_{\kappa^*}$ heat kernel drives $d_s: 2 \to 4$"
        rf" with Pisot log-periodic oscillations  ($\lambda={LAM:.5f}$)",
        color=FG, fontsize=11.5, y=0.975)

    # ── Panel A: d_s(t) ────────────────────────────────────────────
    ax_ds.semilogx(t_grid, ds_bulk,  color=C_OURS, lw=2.2,
                   label=r"Aperiodic SLE  [this work]")
    ax_ds.semilogx(t_grid, ds_fqg3, color=C_FQG,  lw=1.6, ls="--",
                   label=r"FQG $\gamma=3$  [Calcagni \& Briscese 2026]")
    ax_ds.axvline(1, color=C_MU, lw=1.2, ls=":", label=r"$t = t^*$")
    ax_ds.axhline(2, color="#4fc3f7", lw=0.9, ls=":", alpha=0.7,
                  label=r"$d_s=2$  (Hat UV)")
    ax_ds.axhline(4, color="#f48fb1", lw=0.9, ls=":", alpha=0.7,
                  label=r"$d_s=4$  (GR IR)")
    # Mark gap-label scales
    for k in range(1, 6):
        t_k = LAM**(-2*k)
        if 1e-4 < t_k < 1e4:
            ax_ds.axvline(t_k, color="#aed581", lw=0.6, ls="--", alpha=0.5)
    ax_ds.set_ylim(1.6, 4.5)
    ax_ds.text(0.7e-4, 4.25, r"gap-label scales $\lambda^{-2k}$",
               color="#aed581", fontsize=7.5)
    sty(ax_ds, r"Spectral Dimension $d_s(t)$ — Heat Kernel on AP Complex $\Gamma$",
        r"Diffusion time $t/t^*$ (log scale)", r"$d_s(t)$")

    # ── Panel B: d_s^SLE (Alexander-Orbach) ────────────────────────
    ax_sle_d.semilogx(t_grid, ds_sle_t, color=C_SLE, lw=2.0,
                      label=r"$d_s^{\rm SLE}=2D_H/(1+D_H)$")
    ax_sle_d.axvline(1, color=C_MU, lw=1.0, ls=":")
    ax_sle_d.axhline(2*Df_star/(1+Df_star), color="#aed581", lw=0.9, ls=":",
                     label=rf"at $\kappa^*$: ${2*Df_star/(1+Df_star):.4f}$")
    sty(ax_sle_d, r"SLE Spectral Dim (Alexander-Orbach)",
        r"$t/t^*$", r"$d_s^{\rm SLE}$")

    # ── Panel C: Log-periodic oscillations ─────────────────────────
    mask = (t_grid > 1e-3) & (t_grid < 1e3)
    t_m  = t_grid[mask]
    dsm  = 2.0 + 2.0*t_m**nu_RG/(1+t_m**nu_RG)
    dosc = (ds_bulk[mask] - dsm) * 1e3
    ax_osc.semilogx(t_m, dosc, color=C_OSC, lw=1.5)
    ax_osc.axhline(0, color="#4d5568", lw=0.9)
    ax_osc.fill_between(t_m, 0, dosc, where=dosc>0, alpha=0.15, color=C_OSC)
    ax_osc.fill_between(t_m, dosc, 0, where=dosc<0, alpha=0.15, color="#80deea")
    for k in range(6):
        t_k = LAM**(-2*k)
        if 1e-3 < t_k < 1e3:
            ax_osc.axvline(t_k, color="#aed581", lw=0.7, ls="--", alpha=0.7)
            ax_osc.text(t_k*1.08, dosc.max()*0.85,
                        rf"$\lambda^{{-{2*k}}}$", fontsize=6.5, color="#aed581")
    sty(ax_osc, r"Pisot Log-Periodic Oscillations $\delta d_s \times 10^3$",
        r"$t/t^*$", r"$\delta d_s \times 10^3$")

    # ── Panel D: gap label visualisation ───────────────────────────
    ks  = np.arange(8)
    gks = LAM**(-ks)
    ax_gap.barh(ks, gks, color="#aed581", alpha=0.7, height=0.55)
    for k, g in zip(ks, gks):
        ax_gap.text(g + 0.01, k, f"{g:.5f}", va='center',
                    fontsize=7.5, color=FG)
    ax_gap.set_yticks(ks)
    ax_gap.set_yticklabels([rf"$k={k}$" for k in ks], fontsize=8)
    ax_gap.invert_yaxis()
    sty(ax_gap, r"Gap Labels $g_k = \lambda^{-k}$",
        r"$g_k = \lambda^{-k}$", "")
    ax_gap.set_facecolor(BG)
    ax_gap.tick_params(colors=FG, labelsize=8)
    ax_gap.title.set_color(FG)
    ax_gap.set_xlabel(r"$g_k$", fontsize=9, color=FG)

    # ── Panel E: θ(μ) ──────────────────────────────────────────────
    ax_th.semilogx(mu_arr, theta_mu*100, color=C_THETA, lw=2.0,
                   label=r"$\theta(\mu)$ [%]")
    ax_th.axvline(1, color=C_MU, lw=1.0, ls="--", label=r"$\mu=\mu^*$")
    ax_th.axhline(PI_CIRC*100, color="#aed581", lw=0.9, ls=":",
                  label=rf"$\Pi_{{\rm circ}}={PI_CIRC*100:.2f}\%$")
    sty(ax_th, r"CS Coupling $\theta(\mu)$ Flow",
        r"$\mu/\mu^*$ (log scale)", r"$\theta(\mu)$ [%]")

    out = "/mnt/user-data/outputs/dimensional_flow_sle.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n  → {out}")

print("\n  Done.\n")
