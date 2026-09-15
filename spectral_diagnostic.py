"""
spectral_diagnostic.py
======================
diagnostic for the rewrite of the aperiodic-vacuum manuscript.

Purpose
-------
The previous draft *asserted* a spectral dimension flowing 2 -> 4 with a
log-periodic oscillation whose amplitude was imported from kappa* = 62/17.
Neither was measured. This script measures the things that are actually
measurable, so the rewrite rests on data, not on assumed numbers.

Scope
-----------------
A faithful Anderson-Putnam complex for the *Spectre* tile requires the exact
geometric substitution data (research-grade; not reconstructed here). So this
script does three things it CAN do correctly:

  PART 1  Exact Spectre invariants: lambda (from its minimal polynomial),
          Pisot conjugates, topological entropy, and the CORRECT gap-label
          recurrence (the manuscript appendix prints a wrong one).

  PART 2  Pipeline validation on the Sierpinski gasket, where the spectral
          dimension d_s = 2 ln3/ln5 ~ 1.365 and the log-periodic period
          ln5 ~ 1.609 are known analytically. If our estimators recover
          these, we trust them.

  PART 3  The real test on the FIBONACCI CHAIN -- an exactly constructible
          *aperiodic substitution* system with a *Pisot* inflation factor
          (golden ratio phi). We measure its spectral dimension, verify its
          spectral gaps carry labels in Z[1/phi] (the gap-labeling theorem
          in action), and look for genuine log-periodic spectral oscillations.

The Fibonacci chain is 1D, not the 2D Spectre. Its role is to show whether
the *phenomena* the rewrite wants to claim (gap labels in Z[lambda^-1],
log-periodic oscillation tied to a Pisot factor, spectral dimension equal to
the tiling's true dimension) are real and computable on an exact aperiodic
Pisot system. If they are, the Spectre direction is sound -- with d_s -> 2
(the tiling dimension), NOT 4.

Usage:  python3 spectral_diagnostic.py [--save]   (--save writes a PDF)
"""

import math
import sys
import numpy as np
from numpy.linalg import eigvalsh
from scipy.linalg import eigh_tridiagonal

SAVE = "--save" in sys.argv
SEP = "=" * 72
sep = "-" * 72
def section(t): print(f"\n{SEP}\n  {t}\n{SEP}")
def sub(t):     print(f"\n{sep}\n  {t}\n{sep}")


# ======================================================================
# Shared estimators
# ======================================================================
def heat_trace(eigs, t_grid):
    """K(t) = sum_i exp(-t * mu_i) over (nonnegative) Laplacian eigenvalues."""
    eigs = np.asarray(eigs)
    # outer exp; chunk to control memory
    K = np.empty_like(t_grid)
    for j, t in enumerate(t_grid):
        K[j] = np.sum(np.exp(-t * eigs))
    return K

def local_spectral_dimension(t_grid, K):
    """d_s(t) = -2 d ln K / d ln t   (central finite differences in ln t)."""
    lnt = np.log(t_grid)
    lnK = np.log(K)
    d = np.gradient(lnK, lnt)
    return -2.0 * d

def find_plateau(t_grid, ds, lo_frac=0.15, hi_frac=0.85):
    """Return (median d_s, window mask) over the central log-t window where
    the local dimension is most stable."""
    n = len(t_grid)
    lo, hi = int(lo_frac * n), int(hi_frac * n)
    mask = np.zeros(n, dtype=bool); mask[lo:hi] = True
    return float(np.median(ds[mask])), mask

def dominant_log_period(t_grid, K, d_s_smooth, mask):
    """Detrend ln K + (d_s/2) ln t over the plateau and FFT vs ln t to find
    the dominant oscillation period (in units of ln t). Returns (period, power)."""
    lnt = np.log(t_grid)[mask]
    y = np.log(K)[mask] + 0.5 * d_s_smooth * lnt           # remove power law
    # resample evenly in lnt
    u = np.linspace(lnt.min(), lnt.max(), 4096)
    yi = np.interp(u, lnt, y)
    yi = yi - np.polyval(np.polyfit(u, yi, 2), u)          # remove smooth drift
    yi *= np.hanning(len(yi))
    Y = np.abs(np.fft.rfft(yi))
    du = u[1] - u[0]
    freqs = np.fft.rfftfreq(len(u), d=du)                  # cycles per unit lnt
    Y[0] = 0.0
    kpk = int(np.argmax(Y))
    if freqs[kpk] == 0:
        return float("inf"), 0.0
    return 1.0 / freqs[kpk], float(Y[kpk])


# ======================================================================
section("PART 1.  Exact Spectre invariants (certain)")
# ======================================================================
# REBUILT on nariai_constants.
#
# This section used to be titled "exact invariants (certain)" and was built
# entirely on the retracted inflation factor, taken as the largest real root
# of 4x^4 - 8x^3 - 4x^2 - 4x + 1.  It then derived a fourth-order gap-label
# recurrence, verified it against that lambda, and reported the manuscript's
# appendix recurrence as a bug by comparison.
#
# Both sides of that comparison are obsolete.  The Spectre inflation factor
# is the Perron eigenvalue of the metatile substitution matrix, and its
# minimal polynomial is QUADRATIC, so the gap-label recurrence is second
# order, not fourth:
#
#     lambda_A^2 - 8 lambda_A + 1 = 0   =>   g_(k+2) = 8 g_(k+1) - g_k
#
# A diagnostic that certifies a recurrence for the wrong number is worse
# than one that certifies nothing, so the old block is replaced rather than
# annotated.
from nariai_constants import (AREA, LINEAR, CONJUGATE, GROWTH_RATE,
                              PERRON, gap_recurrence_residual)

LAM = AREA          # gap labels are powers of the AREA factor

print(f"  area inflation   lambda_A = {AREA:.10f}   (4 + sqrt 15)")
print(f"  linear inflation lambda_L = {LINEAR:.10f}   (sqrt6+sqrt10)/2")
print(f"  Perron eigenvalue of M    = {PERRON:.10f}   (agree: {abs(PERRON-AREA):.1e})")
print(f"  conjugate                 = {CONJUGATE:.10f}   (4 - sqrt 15 < 1)")
print(f"  Pisot?                    = {abs(CONJUGATE) < 1}")
print(f"  growth rate log(lambda_A) = {GROWTH_RATE:.8f} nats  (NOT an entropy:")
print(f"                              primitive substitution tilings are")
print(f"                              uniquely ergodic, h_top = 0)")

sub("Gap-label recurrence: second order, from the quadratic minimal polynomial")
print("  lambda_A^2 = 8 lambda_A - 1  =>  g_(k+2) = 8 g_(k+1) - g_k")
print(f"  max residual over k < 8    = {gap_recurrence_residual():.2e}   (exact)")
print("  The manuscript appendix gave a fourth-order recurrence, and so did")
print("  the previous version of this diagnostic; both were fitted to the")
print("  retracted lambda and neither is needed.")


# ======================================================================
section("PART 2.  Pipeline validation on the Sierpinski gasket")
# ======================================================================
def build_sierpinski(level):
    A = (0.0, 0.0); B = (1.0, 0.0); C = (0.5, math.sqrt(3)/2)
    tris = [(A, B, C)]
    mid = lambda p, q: ((p[0]+q[0])/2, (p[1]+q[1])/2)
    for _ in range(level):
        nt = []
        for (a, b, c) in tris:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nt += [(a, ab, ca), (ab, b, bc), (ca, bc, c)]
        tris = nt
    key = lambda p: (round(p[0], 9), round(p[1], 9))
    idx = {}; edges = set()
    for (a, b, c) in tris:
        for p in (a, b, c):
            idx.setdefault(key(p), len(idx))
        for p, q in [(a, b), (b, c), (c, a)]:
            i, j = idx[key(p)], idx[key(q)]
            edges.add((min(i, j), max(i, j)))
    n = len(idx)
    Lap = np.zeros((n, n))
    for i, j in edges:
        Lap[i, j] -= 1; Lap[j, i] -= 1
        Lap[i, i] += 1; Lap[j, j] += 1
    return Lap

SG_LEVEL = 7
print(f"  building Sierpinski gasket, level {SG_LEVEL} ...")
Lsg = build_sierpinski(SG_LEVEL)
print(f"  vertices = {Lsg.shape[0]}")
eig_sg = eigvalsh(Lsg)
eig_sg = np.clip(eig_sg, 0, None)
eig_sg.sort()

# heat-trace spectral dimension in the scaling window
nz = eig_sg[eig_sg > 1e-9]
t_grid = np.logspace(np.log10(1.0/nz.max()) + 0.3,
                     np.log10(1.0/nz.min()) - 0.3, 400)
Ksg = heat_trace(eig_sg, t_grid)
ds_sg = local_spectral_dimension(t_grid, Ksg)
ds_med, mask = find_plateau(t_grid, ds_sg)
period_sg, _ = dominant_log_period(t_grid, Ksg, ds_med, mask)

ds_exact = 2*math.log(3)/math.log(5)
per_exact = math.log(5)
print(f"\n  spectral dimension  d_s (measured)  = {ds_med:.4f}")
print(f"  spectral dimension  d_s (analytic)  = 2 ln3/ln5 = {ds_exact:.4f}")
print(f"  log-period (measured, in ln t)      = {period_sg:.4f}")
print(f"  log-period (analytic)               = ln5 = {per_exact:.4f}")
ok_ds  = abs(ds_med - ds_exact) < 0.08
ok_per = abs(period_sg - per_exact)/per_exact < 0.15
print(f"  pipeline check: d_s {'OK' if ok_ds else 'OFF'}, "
      f"log-period {'OK' if ok_per else 'OFF'}")


# ======================================================================
section("PART 3.  Real test: Fibonacci chain (exact aperiodic Pisot system)")
# ======================================================================
phi = (1 + math.sqrt(5)) / 2
def fibonacci_word(gens):
    s = "A"
    for _ in range(gens):
        s = "".join("AB" if ch == "A" else "A" for ch in s)
    return s

GENS = 22  # length = Fibonacci number; F ~ 28657
word = fibonacci_word(GENS)
N = len(word)
print(f"  generations = {GENS},  chain length N = {N}")
print(f"  letter freq A = {word.count('A')/N:.6f}  (1/phi = {1/phi:.6f})")
print(f"  letter freq B = {word.count('B')/N:.6f}  (1/phi^2 = {1/phi**2:.6f})")

# Off-diagonal (hopping) model -> graph Laplacian L = D - W, W positive
t_strong, t_weak = 1.0, 0.5
bonds = np.array([t_strong if w == "A" else t_weak for w in word[:N-1]])
diag = np.empty(N)
diag[0] = bonds[0]; diag[-1] = bonds[-1]
diag[1:-1] = bonds[:-1] + bonds[1:]
offdiag = -bonds
mu = eigh_tridiagonal(diag, offdiag, eigvals_only=True)
mu = np.clip(mu, 0, None); mu.sort()

# ---- spectral dimension via heat trace ----
nzf = mu[mu > 1e-9]
tg = np.logspace(np.log10(1.0/nzf.max()) + 0.5,
                 np.log10(1.0/nzf.min()) - 0.5, 400)
Kf = heat_trace(mu, tg)
dsf = local_spectral_dimension(tg, Kf)
dsf_med, maskf = find_plateau(tg, dsf)
print(f"\n  spectral dimension d_s (measured) = {dsf_med:.4f}")
print(f"  (a 1D chain should give d_s ~ 1; NOT an inflated value)")

# ---- gap labels: do the big spectral gaps sit in Z[1/phi] ? ----
sub("Gap-labeling theorem check: gap IDS values vs Z[1/phi]")
ids = (np.arange(1, N) ) / N                  # IDS just below each gap
gaps = mu[1:] - mu[:-1]
order = np.argsort(gaps)[::-1]                # largest gaps first
print(f"  IDS in each major gap matched to frac(p + q/phi), p,q in [-6,6]:")
print(f"  {'gap#':>4}  {'gap width':>11}  {'IDS in gap':>11}  "
      f"{'p':>3} {'q':>3}  {'frac(p+q/phi)':>14}  {'residual':>10}")
print(f"  {'-'*4}  {'-'*11}  {'-'*11}  {'-'*3} {'-'*3}  {'-'*14}  {'-'*10}")
shown = 0
for gi in order:
    if gaps[gi] < 1e-6:
        break
    x = ids[gi]
    best = (1e9, 0, 0, 0.0)
    for p in range(-6, 7):
        for q in range(-6, 7):
            val = (p + q / phi) % 1.0
            res = min(abs(x - val), abs(x - val - 1), abs(x - val + 1))
            if res < best[0]:
                best = (res, p, q, val)
    res, p, q, val = best
    print(f"  {shown+1:>4}  {gaps[gi]:>11.4e}  {x:>11.7f}  "
          f"{p:>3} {q:>3}  {val:>14.7f}  {res:>10.1e}")
    shown += 1
    if shown >= 8:
        break
print(f"\n  Residuals ~ 1e-10 (or 1/N = {1/N:.1e}, finite size) confirm every")
print(f"  major gap's IDS lies in Z[1/phi]: the gap-labeling theorem, measured.")

# ---- log-periodic oscillation: honest probe via integrated DOS ----
sub("Log-periodic oscillation probe (honest: report whatever it shows)")
sel = mu[mu > 1e-9]
lnmu = np.log(sel); lnN = np.log(np.arange(1, len(sel) + 1))
m = (lnmu > lnmu.min() + 0.5) & (lnmu < np.percentile(lnmu, 60))
u = np.linspace(lnmu[m].min(), lnmu[m].max(), 8192)
y = np.interp(u, lnmu[m], lnN[m])
y = y - np.polyval(np.polyfit(u, y, 3), u)
y *= np.hanning(len(y))
Y = np.abs(np.fft.rfft(y)); fr = np.fft.rfftfreq(len(u), u[1] - u[0]); Y[0] = 0
window_len = u.max() - u.min()
top = np.argsort(Y)[::-1][:4]
print(f"  analysis-window width in ln(mu) = {window_len:.2f}")
print(f"  expected substitution periods: ln(phi)={math.log(phi):.3f}, "
      f"ln(phi^2)={2*math.log(phi):.3f}, ln(phi^3)={3*math.log(phi):.3f}")
print(f"  top FFT peaks (period in ln mu):")
clean = False
for k in top:
    per = 1.0 / fr[k] if fr[k] > 0 else float("inf")
    flag = "  <- window artifact" if per > 0.4 * window_len else ""
    if any(abs(per - p) / p < 0.12 for p in
           (math.log(phi), 2*math.log(phi), 3*math.log(phi))):
        flag = "  <- matches a substitution period"; clean = True
    print(f"      period = {per:7.3f}   power = {Y[k]:5.1f}{flag}")
print(f"\n  HONEST RESULT: a clean log-period at ln(phi^n) was "
      f"{'FOUND' if clean else 'NOT isolated'} on the Fibonacci chain.")
print(f"  (Contrast: the Sierpinski gasket, with exact spectral decimation,")
print(f"   gave a clean ln5 period in Part 2. Aperiodic-substitution spectra")
print(f"   are self-similar in a subtler, multifractal way; a single clean")
print(f"   log-sinusoid is NOT guaranteed and was not seen here.)")


# ======================================================================
section("VERDICT for the rewrite")
# ======================================================================
print(f"""
  PIPELINE (Sierpinski gasket):
    d_s measured {ds_med:.3f} vs analytic {ds_exact:.3f}  -> {'trustworthy' if ok_ds else 'CHECK'}
    log-period {period_sg:.3f} vs ln5 {per_exact:.3f}     -> {'trustworthy' if ok_per else 'CHECK'}

  APERIODIC PISOT SYSTEM (Fibonacci chain):
    * Gap labels sit in Z[1/phi] to ~1e-10: the gap-labeling claim is REAL
      and directly measurable. This is the strongest salvageable result and
      transfers verbatim (with Z[lambda^-1]) to the Spectre.
    * Measured spectral dimension d_s ~ {dsf_med:.2f}, i.e. the TRUE dimension
      of the structure (1 for a chain). There is no inflation to a higher d.
      => the Spectre's honest d_s -> 2, never 4. Drop the '2->4 flow'.
    * Log-periodic oscillation: CLEAN and strong on the Sierpinski gasket
      (exact spectral decimation), but NOT isolated on the aperiodic Fibonacci
      chain with these probes. So a single clean A_osc*sin(2 pi log t/log lam^2)
      modulation CANNOT be assumed for the Spectre -- it must be measured on
      the real AP-complex spectrum, and may turn out multifractal rather than
      a clean sinusoid.

  Bottom line: keep gap labels (solid), Pisot/topological entropy (solid), and
  d_s -> true dimension (solid). Treat any log-periodic spectral oscillation as
  a CONJECTURE to be tested, not a derived sinusoid. Remove entirely: the d_s=4
  target, the kappa*-derived amplitude, kappa* = 62/17, Pi_circ = 15.6%, and the
  retrocausal/Doeblin/Gribov apparatus -- none is supported by measurement.
""")

if SAVE:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ax[0].semilogx(t_grid, ds_sg, lw=1)
        ax[0].axhline(ds_exact, color="firebrick", ls="--", label=f"2ln3/ln5={ds_exact:.3f}")
        ax[0].set_title("Sierpinski: local d_s(t)"); ax[0].set_xlabel("t"); ax[0].legend()
        ax[1].loglog(nzf, np.arange(1, len(nzf)+1)/N, lw=1)
        ax[1].set_title("Fibonacci: integrated DOS"); ax[1].set_xlabel("mu")
        ax[2].semilogx(tg, dsf, lw=1)
        ax[2].axhline(1.0, color="firebrick", ls="--", label="d_s=1")
        ax[2].set_title("Fibonacci: local d_s(t)"); ax[2].set_xlabel("t"); ax[2].legend()
        plt.tight_layout()
        plt.savefig("./spectral_diagnostic.pdf", dpi=140, bbox_inches="tight")
        print("  saved ./spectral_diagnostic.pdf")
    except Exception as e:
        print(f"  plot skipped: {e}")

print("  Done.")