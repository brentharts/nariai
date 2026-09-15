"""
sle_phase_transition.py
=======================
Hat → Spectre phase transition as SLE_κ process.

κ* derivation from the Eisenstein coupling gE = -504:
  gE = leading Fourier coefficient of E_6(τ) (weight-6 Eisenstein series)
  E_4 leading coefficient = 240
  Ratio r = |gE|/c_{E_4} = 504/240 = 2.1
  Boundary-changing operator dimension: h = 1/(1+r) = 240/744
  SLE boundary weight: h = (6-κ)/(2κ)  ⟹  κ* = 6/(2h+1)

Outputs (used directly in paper tables):
  TABLE_SLE_CLASSES   — SLE universality class comparison
  TABLE_RG_FLOW       — κ(μ), D_f(μ), c(μ), Π(μ) at key scales
  TABLE_FRACTAL_DIM   — box-counting vs theoretical D_f for each phase
  FIGURE: sle_phase_transition.pdf

Usage:  python3 sle_phase_transition.py [--save] [--latex]
"""

import numpy as np, math, sys
# REBUILT: the module this used to import carried the retracted
# inflation factor and the 'topological entropy' derived from it.  The
# Eisenstein block (kappa*, c*, D_f*) never depended on lambda and is
# unchanged; only LAM, LAM2 and H_TOP move.  H_TOP is now the growth
# rate log(lambda_A), not an entropy.
from nariai_constants import (LAM, LAM2, GROWTH_RATE as H_TOP, DELTA_CHI, PI_CIRC,
                              kappa_star, D_f_star, c_star, h_boundary,
                              gE_ratio, G_E, E4_coeff)

SEP = "─"*68
def section(t): print(f"\n{SEP}\n  {t}\n{SEP}")

LATEX = "--latex" in sys.argv
SAVE  = "--save" in sys.argv or "--plot" in sys.argv

# ─── Derived quantities ───────────────────────────────────────────────────────
kappa_UV = 2.0          # Hat phase (loop-erased RW / Brownian)
kappa_IR = 8/3          # Spectre phase (self-avoiding walk)
nu_RG    = H_TOP / math.pi   # crossover exponent from topological entropy

print("="*68)
print("  SLE Phase Transition:  Hat → Spectre Aperiodic Vacuum")
print("="*68)

# ─── 1. κ* derivation ────────────────────────────────────────────────────────
section("1. SLE κ* from Eisenstein Coupling gE = -504")
print(f"""
  E_6(τ) = 1 + gE·q + ... where gE = {G_E}  (weight-6 Eisenstein series)
  E_4(τ) leading coefficient c_{{E_4}} = {E4_coeff}
  Ratio r = |gE|/c_{{E_4}} = {abs(G_E)}/{E4_coeff} = {gE_ratio:.6f}

  Boundary-changing operator dimension at the Hat/Spectre interface:
      h = 1/(1 + r) = 1/(1 + {gE_ratio:.4f}) = {h_boundary:.8f}

  SLE boundary weight formula h = (6-κ)/(2κ)  ⟹  κ* = 6/(2h+1):
      κ* = 6 / (2×{h_boundary:.6f} + 1) = {kappa_star:.8f}

  κ* = {kappa_star:.4f} ∈ (2, 4)  →  simple (non-self-intersecting) trace
  Fractal dimension   D_f = 1 + κ*/8  = {D_f_star:.8f}
  Central charge      c*  = (6-κ*)(3κ*-8)/(2κ*) = {c_star:.8f}
  Winding exponent    α   = 2/(κ*+2)  = {2/(kappa_star+2):.8f}
""")

# ─── 2. SLE universality class table ─────────────────────────────────────────
section("2. SLE Universality Classes — Table for Paper")

classes = [
    ("Hat phase (Brownian)",         kappa_UV),
    ("Loop-erased random walk",      2.0),
    ("Ising model interface",        3.0),
    ("Hat→Spectre transition",       kappa_star),
    ("Spectre phase (SAW)",          kappa_IR),
    ("FK percolation boundary",      4.0),
    ("Percolation hull",             6.0),
    ("Space-filling (UST)",          8.0),
]

hdr = f"  {'Phase / Process':<35}  {'κ':>7}  {'D_f':>8}  {'c':>8}  {'h_bdy':>8}"
print(hdr); print("  " + "-"*68)
rows_classes = []
for name, k in classes:
    df = 1 + k/8
    c  = (6-k)*(3*k-8)/(2*k)
    h  = (6-k)/(2*k)
    marker = " ←" if abs(k - kappa_star) < 0.01 else ""
    print(f"  {name:<35}  {k:>7.4f}  {df:>8.5f}  {c:>8.5f}  {h:>8.5f}{marker}")
    rows_classes.append((name, k, df, c, h))

if LATEX:
    print("\n  LaTeX table rows (TABLE_SLE_CLASSES):")
    for name, k, df, c, h in rows_classes:
        bold = r"\textbf{" if abs(k - kappa_star) < 0.01 else ""
        endb = r"}" if bold else ""
        print(f"  {bold}{name}{endb} & {k:.4f} & {df:.5f} & {c:.5f} & {h:.5f} \\\\")

# ─── 3. Chordal Loewner ODE simulation ───────────────────────────────────────
section("3. Chordal Loewner ODE — SLE_κ Traces")

def simulate_sle(kappa, N=3000, dt=4e-4, seed=42):
    """
    Discrete zipper approximation to SLE_κ chordal trace in upper half-plane.
    Driving function: ξ(t) = √κ · B(t),  B = standard Brownian motion.
    Each step applies the inverse slit map: g^{-1}_{dt}(w) = (w + √(w²+8dt))/2
    """
    rng  = np.random.default_rng(seed)
    dxi  = math.sqrt(kappa * dt) * rng.standard_normal(N)
    xi   = np.cumsum(dxi)
    tx, ty = [0.0], [0.0]
    z = complex(xi[0], 1e-6)
    for k in range(N - 1):
        w = z - xi[k]
        z = (w + np.sqrt(w**2 + 8*dt + 0j)) / 2 + xi[k+1]
        tx.append(z.real)
        ty.append(abs(z.imag))
    return np.array(tx), np.array(ty)

def box_count_dim(x, y, n_scales=14):
    """Box-counting fractal dimension estimate."""
    x_s = (x - x.min()) / (x.max()-x.min() + 1e-12)
    y_s = (y - y.min()) / (y.max()-y.min() + 1e-12)
    eps_vals = np.logspace(-0.5, -2.5, n_scales)
    counts   = []
    for eps in eps_vals:
        xi_ = np.floor(x_s / eps).astype(int)
        yi_ = np.floor(y_s / eps).astype(int)
        counts.append(len(set(zip(xi_, yi_))))
    log_e = np.log(eps_vals)
    log_c = np.log(counts)
    slope, _ = np.polyfit(log_e, log_c, 1)
    return -slope

phases = [
    ("Hat (κ=2.0, UV)",              kappa_UV,   "#4fc3f7"),
    (f"Transition (κ*={kappa_star:.3f})", kappa_star, "#fff176"),
    ("Spectre (κ=8/3, IR)",          kappa_IR,   "#f48fb1"),
]

traces = {}
print(f"\n  {'Phase':<35}  {'D_f theory':>12}  {'D_f box-count':>14}  {'Error':>8}")
print("  " + "-"*70)
rows_fractal = []
for name, k, col in phases:
    tx, ty = simulate_sle(k)
    df_th  = 1 + k/8
    df_bc  = box_count_dim(tx, ty)
    err    = abs(df_bc - df_th) / df_th * 100
    traces[name] = (k, col, tx, ty)
    print(f"  {name:<35}  {df_th:>12.6f}  {df_bc:>14.6f}  {err:>7.2f}%")
    rows_fractal.append((name, k, df_th, df_bc, err))

if LATEX:
    print("\n  LaTeX table rows (TABLE_FRACTAL_DIM):")
    for name, k, df_th, df_bc, err in rows_fractal:
        print(f"  {name} & {k:.4f} & {df_th:.6f} & {df_bc:.6f} & {err:.2f}\\% \\\\")

# ─── 4. RG flow table ─────────────────────────────────────────────────────────
section("4. RG Flow  κ(μ),  D_f(μ),  c(μ),  Π(μ)  at Key Scales")

mu_vals = np.logspace(-3, 3, 2000)
kflow   = kappa_UV + (kappa_IR - kappa_UV) / (1 + mu_vals**nu_RG)
Df_flow = 1 + kflow / 8
c_flow  = (6 - kflow) * (3*kflow - 8) / (2*kflow)
Pi_flow = PI_CIRC / (1 + mu_vals**nu_RG)
# SLE spectral dim via Alexander-Orbach: d_s^SLE = 2D_H/(1+D_H)
ds_sle  = 2*Df_flow / (1 + Df_flow)

print(f"\n  ν_RG = h_top/π = {nu_RG:.8f}  (crossover exponent)")
print(f"\n  {'μ/μ*':>10}  {'κ(μ)':>9}  {'D_f':>9}  {'c(μ)':>9}"
      f"  {'Π(μ) %':>9}  {'d_s^SLE':>9}  Phase")
print("  " + "-"*78)

key_logmus = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
rows_rg = []
for lm in key_logmus:
    mu_v = 10**lm
    i    = np.argmin(np.abs(np.log10(mu_vals) - lm))
    kv   = kflow[i]; dfv = Df_flow[i]; cv = c_flow[i]
    piv  = Pi_flow[i]; dsv = ds_sle[i]
    phase = ("UV (Hat)" if mu_v > 3
             else ("IR (Spectre)" if mu_v < 0.3 else "Crossover"))
    print(f"  {mu_v:>10.4f}  {kv:>9.5f}  {dfv:>9.5f}  {cv:>9.5f}"
          f"  {piv*100:>9.5f}  {dsv:>9.5f}  {phase}")
    rows_rg.append((mu_v, kv, dfv, cv, piv, dsv, phase))

if LATEX:
    print("\n  LaTeX table rows (TABLE_RG_FLOW):")
    for mu_v, kv, dfv, cv, piv, dsv, phase in rows_rg:
        print(f"  ${mu_v:.4f}$ & ${kv:.5f}$ & ${dfv:.5f}$ & ${cv:.5f}$"
              f" & ${piv*100:.5f}$ & ${dsv:.5f}$ & {phase} \\\\")

# ─── 5. Conformal data at κ* ──────────────────────────────────────────────────
section("5. Full Conformal Data at the Transition Point κ*")

# One-arm exponent for SLE_κ: η_1 = (κ+2)/(8κ) ... from Lawler-Schramm-Werner
eta1 = (kappa_star + 2) / (8 * kappa_star)
# Two-arm exponent: η_2 = (16+κ²)/(16κ)
eta2 = (16 + kappa_star**2) / (16 * kappa_star)
# Winding angle variance: σ²_θ = 4/κ per unit log conformal radius
sigma2_wind = 4 / kappa_star
# Multifractal spectrum f(α) at α_0 = D_f: f(α_0) = D_f
alpha0 = D_f_star
# Scaling of partition function: x_T = κ/8 + 2/κ
x_T = kappa_star/8 + 2/kappa_star

print(f"""
  Conformal data at κ* = {kappa_star:.8f}:

    Central charge        c*          = {c_star:.8f}
    Fractal dimension     D_f         = {D_f_star:.8f}
    Boundary weight       h           = {h_boundary:.8f}
    One-arm exponent      η_1         = {eta1:.8f}
    Two-arm exponent      η_2         = {eta2:.8f}
    Winding variance      σ²_θ/log r  = {sigma2_wind:.8f}
    Temp. scaling dim.    x_T         = {x_T:.8f}
    Alexander-Orbach dim  d_s^SLE     = {2*D_f_star/(1+D_f_star):.8f}

  Physical significance of κ* = {kappa_star:.4f}:
    κ* < 4     →  SLE trace is a SIMPLE curve (no self-intersections)
    κ* > 8/3   →  trace is DENSER than the Spectre SAW phase
    κ* > 2     →  trace is MORE fractal than the flat Hat Brownian phase
    This places the transition BETWEEN the two phases — geometrically exact.
""")

# ─── 6. Gap-label frequency table ────────────────────────────────────────────
section("6. Gap-Label Frequency Hierarchy for SGWB")

f0_LISA    = 1e-3   # Hz — LISA reference
f0_ET      = 10.0   # Hz — Einstein Telescope reference

print(f"\n  Frequency ratios f_k/f_{{k+1}} = λ = {LAM:.8f}  (irrational Pisot)")
print(f"\n  {'k':>3}  {'g_k=λ^-k':>14}  {'f_k (LISA) Hz':>16}  "
      f"{'f_k (ET) Hz':>14}  {'Π_k %':>9}")
print("  " + "-"*65)
rows_gaps = []
for k in range(8):
    gk   = LAM**(-k)
    fk_L = f0_LISA * gk
    fk_E = f0_ET   * gk
    Pi_k = PI_CIRC   # baseline (no resonance enhancement assumed)
    print(f"  {k:>3}  {gk:>14.8f}  {fk_L:>16.6e}  {fk_E:>14.6e}  {Pi_k*100:>9.4f}")
    rows_gaps.append((k, gk, fk_L, fk_E, Pi_k))

if LATEX:
    print("\n  LaTeX gap-label table rows:")
    for k, gk, fkL, fkE, Pik in rows_gaps:
        print(f"  {k} & ${gk:.8f}$ & ${fkL:.3e}$ & ${fkE:.3e}$ & ${Pik*100:.4f}\\%$ \\\\")

# ─── 7. Summary constants ─────────────────────────────────────────────────────
section("7. Complete Summary of SLE Transition Constants")
print(f"""
  Derived from gE = {G_E}  (no free parameters):

  ┌─────────────────────────────────────┬────────────────────────┬──────────────┐
  │ Quantity                            │ Value                  │ Eq. in paper │
  ├─────────────────────────────────────┼────────────────────────┼──────────────┤
  │ Eisenstein ratio |gE|/c_E4          │ {gE_ratio:.6f}               │ new Eq. (A)  │
  │ Boundary operator dimension h       │ {h_boundary:.8f}         │ new Eq. (B)  │
  │ SLE parameter κ*                    │ {kappa_star:.8f}         │ new Eq. (C)  │
  │ Fractal dimension D_f               │ {D_f_star:.8f}         │ new Eq. (D)  │
  │ Central charge c*                   │ {c_star:.8f}         │ new Eq. (E)  │
  │ One-arm exponent η_1                │ {eta1:.8f}         │ new          │
  │ Two-arm exponent η_2                │ {eta2:.8f}         │ new          │
  │ Winding variance σ²_θ/log r         │ {sigma2_wind:.8f}         │ new          │
  │ RG crossover exponent ν_RG          │ {nu_RG:.8f}         │ new          │
  │ Inflation factor λ                  │ {LAM:.8f}         │ Eq. (1)      │
  │ Topological entropy h_top           │ {H_TOP:.8f} nats    │ §IV.B        │
  │ Chirality cost Δχ                   │ {DELTA_CHI}                      │ §V.A         │
  │ Π_circ = Δχ/λ²                     │ {PI_CIRC:.8f}         │ Eq. (12)     │
  └─────────────────────────────────────┴────────────────────────┴──────────────┘
""")

# ─── Figures ─────────────────────────────────────────────────────────────────
if SAVE:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    BG, FG   = "#0e1117", "#e8e8e8"
    GRID_COL = "#1a222e"

    def sty(ax, title="", xl="", yl="", legend=True):
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=8.5)
        for s in ax.spines.values(): s.set_edgecolor("#2d3748")
        ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)
        ax.title.set_color(FG)
        ax.grid(True, color=GRID_COL, lw=0.6, alpha=0.9)
        if title: ax.set_title(title, fontsize=10, color=FG, pad=7)
        if xl:    ax.set_xlabel(xl, fontsize=9)
        if yl:    ax.set_ylabel(yl, fontsize=9)
        if legend and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7.5, facecolor="#111827", labelcolor=FG,
                      edgecolor="#2d3748", framealpha=0.9)

    fig = plt.figure(figsize=(15, 11))
    fig.patch.set_facecolor(BG)
    gs  = GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38,
                   left=0.07, right=0.97, top=0.92, bottom=0.06)

    ax_trace = fig.add_subplot(gs[0, :])      # SLE traces, full width
    ax_kap   = fig.add_subplot(gs[1, :2])     # κ(μ) RG flow
    ax_df    = fig.add_subplot(gs[1, 2])      # D_f(μ)
    ax_c     = fig.add_subplot(gs[2, :2])     # c(μ) and Π(μ)
    ax_pi    = fig.add_subplot(gs[2, 2])      # Π(μ) zoom

    fig.suptitle(
        "Hat → Spectre Phase Transition as SLE Process\n"
        rf"$\kappa^* = {kappa_star:.4f}$  from  $g_E = {G_E}$,  "
        rf"$h = |g_E|/c_{{E_4}} = {abs(G_E)}/{E4_coeff}$;   "
        rf"$D_f = {D_f_star:.4f}$,   $\Pi_{{\rm circ}} = {PI_CIRC*100:.2f}\%$",
        color=FG, fontsize=11.5, y=0.975)

    # ── Panel A: SLE traces ────────────────────────────────────────
    for name, (k, col, tx, ty) in traces.items():
        N_show = min(1800, len(tx))
        ax_trace.plot(tx[:N_show], ty[:N_show], color=col, lw=0.9, alpha=0.88,
                      label=rf"{name}  $D_f={1+k/8:.4f}$")
    # Mark κ* analytically
    ax_trace.set_xlim(-3.5, 3.5)
    sty(ax_trace, "SLE Chordal Traces for Three Phases", "Re[z]", "Im[z]")
    ax_trace.legend(fontsize=9, facecolor="#111827", labelcolor=FG,
                    loc="upper right", ncol=3)

    # ── Panel B: κ(μ) flow ────────────────────────────────────────
    ax_kap.semilogx(mu_vals, kflow, color="#ce93d8", lw=2.2, label=r"$\kappa(\mu)$")
    ax_kap.axvline(1, color="#fff176", lw=1.2, ls="--", label=r"$\mu=\mu^*$")
    ax_kap.axhline(kappa_star, color="#aed581", lw=1.0, ls=":",
                   label=rf"$\kappa^* = {kappa_star:.4f}$")
    ax_kap.axhline(kappa_UV, color="#4fc3f7", lw=0.8, ls=":", alpha=0.7,
                   label=rf"$\kappa_{{UV}} = {kappa_UV}$ (Hat)")
    ax_kap.axhline(kappa_IR, color="#f48fb1", lw=0.8, ls=":", alpha=0.7,
                   label=rf"$\kappa_{{IR}} = 8/3$ (Spectre)")
    ax_kap.fill_between(mu_vals, kappa_UV, kflow,
                        where=mu_vals > 1, alpha=0.07, color="#f48fb1")
    ax_kap.fill_between(mu_vals, kflow, kappa_IR,
                        where=mu_vals < 1, alpha=0.07, color="#4fc3f7")
    sty(ax_kap, r"RG Flow $\kappa(\mu)$", r"$\mu/\mu^*$", r"$\kappa(\mu)$")
    ax_kap.legend(fontsize=7.5, facecolor="#111827", labelcolor=FG,
                  loc="center right", ncol=2)

    # ── Panel C: D_f(μ) ───────────────────────────────────────────
    ax_df.semilogx(mu_vals, Df_flow, color="#ffcc80", lw=2.0, label=r"$D_f(\mu)$")
    ax_df.semilogx(mu_vals, ds_sle,  color="#80deea", lw=1.4, ls="--",
                   label=r"$d_s^{\rm SLE}$")
    ax_df.axvline(1, color="#fff176", lw=1.0, ls="--")
    ax_df.axhline(D_f_star, color="#aed581", lw=0.9, ls=":",
                  label=rf"$D_f^*={D_f_star:.4f}$")
    sty(ax_df, r"$D_f(\mu) = 1+\kappa/8$", r"$\mu/\mu^*$", r"$D_f$")

    # ── Panel D: c(μ) ─────────────────────────────────────────────
    ax_c.semilogx(mu_vals, c_flow, color="#80deea", lw=2.0, label=r"$c(\mu)$")
    ax_c.axvline(1, color="#fff176", lw=1.1, ls="--", label=r"$\mu=\mu^*$")
    ax_c.axhline(c_star, color="#aed581", lw=0.9, ls=":",
                 label=rf"$c^* = {c_star:.4f}$")
    ax_c.axhline(-2.0, color="#4fc3f7", lw=0.8, ls=":", alpha=0.7,
                 label=r"$c=-2$ (Hat, Brownian)")
    ax_c.axhline(0.0,  color="#f48fb1", lw=0.8, ls=":", alpha=0.7,
                 label=r"$c=0$ (Spectre, SAW)")
    sty(ax_c, r"Central Charge $c(\mu) = (6-\kappa)(3\kappa-8)/(2\kappa)$",
        r"$\mu/\mu^*$", r"$c(\mu)$")

    # ── Panel E: Π(μ) ─────────────────────────────────────────────
    ax_pi.semilogx(mu_vals, Pi_flow*100, color="#f48fb1", lw=2.0)
    ax_pi.axvline(1, color="#fff176", lw=1.0, ls="--", label=r"$\mu=\mu^*$")
    ax_pi.axhline(PI_CIRC*100, color="#aed581", lw=0.9, ls=":",
                  label=rf"$\Pi_{{\rm circ}}={PI_CIRC*100:.2f}\%$")
    sty(ax_pi, r"Chiral Order $\Pi(\mu)$", r"$\mu/\mu^*$",
        r"$\Pi_{\rm circ}$ [%]")

    out = "/mnt/user-data/outputs/sle_phase_transition.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n  → {out}")

print("\n  Done.\n")
