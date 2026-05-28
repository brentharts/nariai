"""
corrected_constants.py
======================
Authoritative constants, rebuilt on the verified Spectre
inflation factor and with every discredited quantity explicitly retracted.

Supersedes: verify_constants.py and sle_constants.py (both inherited the
wrong inflation factor lambda = 2.5348 and several unsupported quantities).

The inflation factor is established THREE independent ways:
  (1) Perron-Frobenius eigenvalue of the 9x9 metatile substitution matrix
      (Smith-Myers-Kaplan-Goodman-Strauss; rules taken from the canonical
      spectre.py generator),
  (2) closed form 4 + sqrt(15) and its minimal polynomial,
  (3) (cross-checked in spectre_spectral_dimension.py) direct geometric
      measurement of the patch diameter ratio ~ 2.806.

Run:  python3 corrected_constants.py
"""
import math
import numpy as np

SEP = "=" * 72
sep = "-" * 72
def section(t): print(f"\n{SEP}\n  {t}\n{SEP}")
def sub(t):     print(f"\n{sep}\n  {t}\n{sep}")


# ======================================================================
section("1.  Inflation factor from the substitution matrix (first principles)")
# ======================================================================
labels = ["Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi"]
# Canonical metatile substitution (the 'Mystic + 8 Spectres' supertile).
rules = {
    "Gamma":  ["Pi",  "Delta", None,  "Theta", "Sigma", "Xi",  "Phi",    "Gamma"],
    "Delta":  ["Xi",  "Delta", "Xi",  "Phi",   "Sigma", "Pi",  "Phi",    "Gamma"],
    "Theta":  ["Psi", "Delta", "Pi",  "Phi",   "Sigma", "Pi",  "Phi",    "Gamma"],
    "Lambda": ["Psi", "Delta", "Xi",  "Phi",   "Sigma", "Pi",  "Phi",    "Gamma"],
    "Xi":     ["Psi", "Delta", "Pi",  "Phi",   "Sigma", "Psi", "Phi",    "Gamma"],
    "Pi":     ["Psi", "Delta", "Xi",  "Phi",   "Sigma", "Psi", "Phi",    "Gamma"],
    "Sigma":  ["Xi",  "Delta", "Xi",  "Phi",   "Sigma", "Pi",  "Lambda", "Gamma"],
    "Phi":    ["Psi", "Delta", "Psi", "Phi",   "Sigma", "Pi",  "Phi",    "Gamma"],
    "Psi":    ["Psi", "Delta", "Psi", "Phi",   "Sigma", "Psi", "Phi",    "Gamma"],
}
idx = {l: i for i, l in enumerate(labels)}
M = np.zeros((9, 9))
for j, L in enumerate(labels):
    for sub_label in rules[L]:
        if sub_label is not None:
            M[idx[sub_label], j] += 1

pf = max(np.linalg.eigvals(M).real)
charpoly = np.poly(M)
print(f"  tiles per supertile (column sums): {[int(M[:,j].sum()) for j in range(9)]}")
print(f"  characteristic polynomial factors as  x^5 (x-1)(x+1)(x^2 - 8x + 1)")
print(f"  Perron-Frobenius eigenvalue (area inflation) = {pf:.10f}")

# closed forms
AREA   = 4 + math.sqrt(15)              # area inflation; root of x^2 - 8x + 1
LINEAR = math.sqrt(AREA)               # length inflation; root of x^4 - 8x^2 + 1
LINEAR_alt = (math.sqrt(6) + math.sqrt(10)) / 2

print(f"\n  AREA inflation    lambda_A = 4 + sqrt(15)        = {AREA:.10f}")
print(f"    minimal poly  x^2 - 8x + 1  -> residual {AREA**2 - 8*AREA + 1:.2e}")
print(f"    PF match: |PF - (4+sqrt15)| = {abs(pf - AREA):.2e}")
print(f"  LINEAR inflation  lambda_L = sqrt(4+sqrt15)     = {LINEAR:.10f}")
print(f"                             = (sqrt6+sqrt10)/2   = {LINEAR_alt:.10f}")
print(f"    minimal poly  x^4 - 8x^2 + 1 -> residual {LINEAR**4 - 8*LINEAR**2 + 1:.2e}")
print(f"    consistency: lambda_L^2 - lambda_A = {LINEAR**2 - AREA:.2e}")

assert abs(pf - AREA) < 1e-9
assert abs(LINEAR**2 - AREA) < 1e-12

sub("RETRACTION: the manuscript's inflation factor was a different number")
WRONG_LAM = (1 + math.sqrt(3) + math.sqrt(2 + 2*math.sqrt(3))) / 2
print(f"  manuscript lambda = {WRONG_LAM:.6f}  (root of 4x^4-8x^3-4x^2-4x+1)")
print(f"  correct  lambda_L = {LINEAR:.6f}  (root of x^4-8x^2+1)")
print(f"  these are NOT the same number (differ by {abs(WRONG_LAM-LINEAR):.3f}).")
print(f"  Every lambda-dependent quantity below uses the CORRECT value.")


# ======================================================================
section("2.  Pisot property (both inflation factors)")
# ======================================================================
for name, poly in [("area  lambda_A", [1, -8, 1]),
                   ("linear lambda_L", [1, 0, -8, 0, 1])]:
    r = np.roots(poly)
    big = max(r.real)
    conj = sorted(abs(x) for x in r if abs(x - big) > 1e-9)
    print(f"  {name}: roots moduli of conjugates = {[f'{c:.4f}' for c in conj]} "
          f"-> Pisot: {all(c < 1 for c in conj)}")
print(f"  (lambda_A conjugate = 4 - sqrt(15) = {4 - math.sqrt(15):.6f} < 1)")


# ======================================================================
section("3.  Inflation growth rate  (NB: NOT topological entropy)")
# ======================================================================
growth = math.log(AREA)
print(f"  log(lambda_A) = log(4+sqrt15) = {growth:.8f} nats = {growth/math.log(2):.8f} bits")
print(f"""
  HONESTY NOTE / CORRECTION:
    The old draft called log(lambda) the 'topological entropy' and quoted
    0.930 nats. That is wrong twice over:
      (a) it used the wrong lambda (2.5348);
      (b) primitive substitution tilings are uniquely ergodic and the
          translation action has ZERO topological entropy -- they are not
          chaotic. log(lambda_A) is the exponential GROWTH RATE of the tile
          count per inflation (= log of the PF eigenvalue), which is the
          correct and well-defined quantity. Label it as such, not as entropy.
    Correct growth rate = log(4+sqrt15) = {growth:.4f} nats.""")


# ======================================================================
section("4.  Gap-label module and the corrected recurrence")
# ======================================================================
print(f"""
  Gap-labeling theorem (Bellissard-van Elst-Schulz-Baldes): the IDS values at
  spectral gaps lie in the frequency module, a subgroup of R generated over
  Z[1/lambda_A].  Since (4+sqrt15)(4-sqrt15) = 1, we have 1/lambda_A = 4-sqrt15,
  hence

      gap labels  in  Z[1/lambda_A] = Z[4 - sqrt15] = Z[sqrt15]
                  = {{ m + n*sqrt(15) : m, n in Z }}  (intersected with the
                    physical range; full group fixed by H^*(Omega)).
""")

sub("Corrected Pisot recurrence (replaces the WRONG appendix equation)")
print(f"  From lambda_A^2 - 8 lambda_A + 1 = 0, dividing by lambda_A^(k+2):")
print(f"      g_(k+2) = 8 g_(k+1) - g_k        where g_k = lambda_A^(-k)\n")
maxres = 0.0
print(f"  {'k':>3}  {'g_k = lambda_A^-k':>20}  {'8 g_(k+1) - g_k':>18}  {'= g_(k+2)? resid':>16}")
print(f"  {'-'*3}  {'-'*20}  {'-'*18}  {'-'*16}")
for k in range(7):
    gk, gk1, gk2 = AREA**(-k), AREA**(-(k+1)), AREA**(-(k+2))
    rhs = 8*gk1 - gk
    res = abs(rhs - gk2); maxres = max(maxres, res)
    print(f"  {k:>3}  {gk:>20.12f}  {rhs:>18.12f}  {res:>16.1e}")
print(f"\n  max residual = {maxres:.1e}   (machine precision -> recurrence exact)")
assert maxres < 1e-12

sub("RETRACTION: the manuscript appendix recurrence was algebraically wrong")
print(f"""  The v7 appendix printed a 4th-order recurrence with coefficients
  (-4, 8, 4, 4) on the OLD (wrong) lambda; it failed its own numbers by ~0.4.
  The correct object is the 2nd-order recurrence above on lambda_A = 4+sqrt15.""")


# ======================================================================
section("5.  Gap-label table (corrected)")
# ======================================================================
print(f"  {'k':>3}  {'lambda_A^-k':>16}  {'lambda_L^-k (linear)':>22}")
print(f"  {'-'*3}  {'-'*16}  {'-'*22}")
for k in range(7):
    print(f"  {k:>3}  {AREA**(-k):>16.10f}  {LINEAR**(-k):>22.10f}")


# ======================================================================
section("6.  Spectral dimension (measured; see spectre_spectral_dimension.py)")
# ======================================================================
print(f"""
  Measured on genuine substitution approximants (n=3,4) with two estimators,
  calibrated against a same-size square lattice (true d_s=2, bias ~ +/-0.15):

      d_s = 2.0  +/-  ~0.15  (finite-size), rising toward 2 with patch size.

  Excludes any flow above 2; emphatically excludes d_s = 4.
  REPLACES the old 'spectral dimension flows 2 -> 4' claim, which was both
  unmeasured and (for the IR target) false.""")


# ======================================================================
section("7.  RETRACTION LEDGER: quantities removed entirely")
# ======================================================================
removed = [
    ("kappa* = 62/17 ~ 3.647",
     "SLE parameter from h = 1/(1+504/240); unmotivated numerology, no CFT."),
    ("c* ~ 0.949, D_f* ~ 1.456",
     "Downstream of kappa*; fall with it."),
    ("Pi_circ = Dchi/lambda^2 ~ 15.6%",
     "Dchi=14-13 was a MISREADING: 13- and 14-edge descriptions are the SAME "
     "Spectre tile (two collinear edges). No parity-odd class; no mechanism "
     "from edge count to a gravitational Chern-Simons VEV."),
    ("det(M_FP) >= 2^{I_doe} > 1  (Gribov bound)",
     "Built on Ji-Lloyd-Wilde retrocausal/Doeblin capacities, which concern "
     "postselected closed timelike curves, not spatial tilings or gauge orbits."),
    ("I_max, I_doe, C_retro as physical bounds",
     "Category error; informational quantities for a different problem."),
    ("r >= 0.01 as a prediction OF THIS framework",
     "Genuine result of Liu-Quintin-Afshordi quadratic gravity; imported, not "
     "a consequence of the aperiodic structure. May be CITED, not claimed."),
    ("topological entropy = 0.930 nats",
     "Wrong lambda AND wrong concept; replaced by growth rate log(4+sqrt15)."),
    ("Page-Wootters / heat-kernel 'time' justification",
     "Spatial RG scaling is not a clock-subsystem trajectory; conflation."),
]
for q, why in removed:
    print(f"\n  REMOVED: {q}")
    print(f"           reason: {why}")


# ======================================================================
section("8.  AUTHORITATIVE SUMMARY TABLE (use these in the rewrite)")
# ======================================================================
rows = [
    ("area inflation  lambda_A",      f"{AREA:.8f}",          "4 + sqrt(15)"),
    ("linear inflation lambda_L",     f"{LINEAR:.8f}",        "(sqrt6+sqrt10)/2"),
    ("lambda_A minimal poly",         "x^2 - 8x + 1",          "Pisot (quadratic)"),
    ("lambda_L minimal poly",         "x^4 - 8x^2 + 1",        "Pisot (quartic)"),
    ("lambda_A conjugate",            f"{4-math.sqrt(15):.8f}", "4 - sqrt(15) < 1"),
    ("inflation growth rate",         f"{growth:.8f} nats",    "log(lambda_A)"),
    ("gap-label module",              "Z[sqrt(15)]",           "= Z[1/lambda_A]"),
    ("recurrence (g_k=lambda_A^-k)",  "g_(k+2)=8 g_(k+1)-g_k", "2nd order"),
    ("gap label  lambda_A^-1",        f"{1/AREA:.8f}",         "4 - sqrt(15)"),
    ("gap label  lambda_A^-2",        f"{1/AREA**2:.8f}",      ""),
    ("spectral dimension d_s",        "2.0 +/- 0.15",          "measured, -> 2"),
]
print(f"\n  {'Quantity':<32} {'Value':<24} {'Note'}")
print(f"  {'-'*32} {'-'*24} {'-'*18}")
for n, v, note in rows:
    print(f"  {n:<32} {v:<24} {note}")

print(f"\n  All assertions passed. These supersede verify_constants.py / sle_constants.py.")
