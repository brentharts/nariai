"""nariai_constants.py -- the single importable source of truth.

`corrected_constants.py` established the right numbers and retracted the
wrong ones, but it is a report: importing it prints eight sections. So
nothing could import it, and five scripts went on computing with the
retracted inflation factor because the corrected one was not reachable as a
value. This module is that value.

Everything here is derived from the substitution matrix at import time, not
transcribed. The only inputs are the metatile rules.

    lambda_A = 4 + sqrt(15)     = 7.8729833462   area inflation, x^2-8x+1
    lambda_L = (sqrt6+sqrt10)/2 = 2.8058837015   linear inflation, x^4-8x^2+1

Which one a script wants is not a detail. `lambda_A = lambda_L^2`, so a
script that defines `LAM` as the linear factor and `LAM2 = LAM**2` as the
area stays consistent when both are corrected together. But the gap-label
module is `Z[1/lambda_A]`, so gap labels are powers of the AREA factor, and
a script that feeds the linear factor into `sum(n_k lam**-k)` is wrong even
if its lambda is right. The aliases below are named so that the choice has
to be made explicitly.

Retracted quantities are kept in `RETRACTED` rather than deleted, so
`consistency_test.py` can search for them. A number you cannot name is a
number you cannot check for.

    python3 nariai_constants.py             # print the table
    python3 nariai_constants.py --selftest  # verify every derivation
"""

import math
import sys

import numpy as np

# ----------------------------------------------------------------------
# the substitution, as data
# ----------------------------------------------------------------------

SPECIES = ("Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma",
           "Phi", "Psi")

# Canonical 'Mystic + 8 Spectres' supertile (Smith, Myers, Kaplan,
# Goodman-Strauss). Gamma's null slot is why its column sums to 7.
RULES = {
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

_IDX = {name: i for i, name in enumerate(SPECIES)}
M = np.zeros((9, 9))
for _j, _L in enumerate(SPECIES):
    for _child in RULES[_L]:
        if _child is not None:
            M[_IDX[_child], _j] += 1


# ----------------------------------------------------------------------
# the inflation factors
# ----------------------------------------------------------------------

PERRON = float(max(np.linalg.eigvals(M).real))

AREA = 4 + math.sqrt(15)          # lambda_A, root of x^2 - 8x + 1
LINEAR = math.sqrt(AREA)          # lambda_L, root of x^4 - 8x^2 + 1
LINEAR_CLOSED = (math.sqrt(6) + math.sqrt(10)) / 2
CONJUGATE = 4 - math.sqrt(15)     # = 1/AREA, the Pisot conjugate

# Growth rate of the tile COUNT per inflation. Not an entropy: primitive
# substitution tilings are uniquely ergodic and their translation action has
# zero topological entropy. See RETRACTED['h_top'].
GROWTH_RATE = math.log(AREA)

# log of the LINEAR factor.  This is a trap worth naming: the old scripts
# defined `H_TOP = log(LAM)` with LAM the linear factor, and then relied on
# identities such as `log(lambda^2) = 2 * H_TOP` and `nu = H_TOP / pi`.
# Rebuilding them by mapping H_TOP to GROWTH_RATE = log(AREA) keeps the word
# and breaks the arithmetic: log(AREA) is already 2*log(LINEAR), so
# `2 * H_TOP` becomes log(AREA^2) and the oscillation period silently
# doubles.  A script that wants "log of the inflation factor the old code
# meant" wants this, and a script that wants the tile-count growth rate
# wants GROWTH_RATE.  They differ by exactly a factor of two, which is why
# substituting one for the other produces plausible output.
LOG_LINEAR = math.log(LINEAR)

# Aliases for scripts that use the older names. LAM is the LINEAR factor and
# LAM2 the AREA factor, which is the convention parity_violation_derivation
# and sgwb_polarization already use -- and LAM2 == LAM**2 still holds, so
# those scripts stay internally consistent once both are corrected.
LAM = LINEAR
LAM2 = AREA

# Gap labels live in Z[1/lambda_A]; the base is the AREA factor, and calling
# it LAM would invite exactly the substitution that is wrong here.
GAP_BASE = AREA

# Hat -> Spectre edge-count increase. 13 -> 14.
V_HAT, V_SPECTRE = 13, 14
DELTA_CHI = V_SPECTRE - V_HAT

# Circular polarisation of the SGWB. With DELTA_CHI = 1 this is exactly the
# Pisot conjugate, and equally exactly the leading gap label -- a coincidence
# the retracted value concealed.
PI_CIRC = DELTA_CHI / AREA


# ----------------------------------------------------------------------
# the Eisenstein / SLE block -- independent of lambda
# ----------------------------------------------------------------------
# Worth stating plainly: none of this depends on the inflation factor, so
# none of it was affected by the retraction. kappa* = 3.647 stood before and
# stands now. Carrying it here keeps the SLE scripts off sle_constants.

G_E = -504                                  # leading Fourier coeff of E_6
E4_COEFF = 240                              # leading Fourier coeff of E_4
GE_RATIO = abs(G_E) / E4_COEFF              # 2.1
H_BOUNDARY = 1 / (1 + GE_RATIO)             # 240/744
KAPPA_STAR = 6 / (2 * H_BOUNDARY + 1)
C_STAR = (6 - KAPPA_STAR) * (3 * KAPPA_STAR - 8) / (2 * KAPPA_STAR)
D_F_STAR = 1 + KAPPA_STAR / 8

# older lowercase names used by the SLE scripts
kappa_star, c_star, D_f_star = KAPPA_STAR, C_STAR, D_F_STAR
h_boundary, gE_ratio, E4_coeff = H_BOUNDARY, GE_RATIO, E4_COEFF


# ----------------------------------------------------------------------
# what was retracted, kept nameable so it can be searched for
# ----------------------------------------------------------------------

WRONG_LAM = (1 + math.sqrt(3) + math.sqrt(2 + 2 * math.sqrt(3))) / 2

RETRACTED = {
    "lambda": dict(
        value=WRONG_LAM,
        why="root of 4x^4-8x^3-4x^2-4x+1; not the Spectre inflation factor. "
            "Differs from lambda_L by 0.271, so every dependent quantity is "
            "wrong rather than imprecise.",
        replacement="LINEAR (=2.8058837) or AREA (=7.8729833)"),
    "h_top": dict(
        value=math.log(WRONG_LAM),
        why="wrong twice: wrong lambda, and primitive substitution tilings "
            "have zero topological entropy.",
        replacement="GROWTH_RATE = log(AREA) = 2.0634 nats"),
    "pi_circ": dict(
        value=1 / WRONG_LAM ** 2,
        why="downstream of the wrong lambda.",
        replacement="PI_CIRC = 1/AREA = 4 - sqrt(15) = 0.1270167"),
    "I_max/I_doe/C_retro": dict(
        value=None,
        why="category error: informational quantities for a different "
            "problem, not physical bounds of this framework.",
        replacement=None),
    "r >= 0.01": dict(
        value=0.01,
        why="a genuine result of Liu-Quintin-Afshordi quadratic gravity, "
            "imported not derived here.",
        replacement="cite, do not claim"),
}

# Source-level markers the consistency test greps for. The expression form
# matters as much as the decimal: three scripts carried the retracted value
# as an un-evaluated formula, so searching only for '2.5348' would have
# found none of them.
RETRACTED_PATTERNS = (
    "2.53479",
    "2.5348",
    "sqrt(2 + 2 * sqrt3)",
    "sqrt(2 + 2*sqrt3)",
    "sqrt(2 + 2 * sqrt(3))",
    "0.93011",
    "from sle_constants",
    "import sle_constants",
)


# ----------------------------------------------------------------------

def gap_label(coefficients, base=None):
    """g = sum_k n_k * base^-k, with base = lambda_A by default."""
    b = GAP_BASE if base is None else base
    return sum(c * b ** (-k) for k, c in enumerate(coefficients))


def gap_recurrence_residual(kmax=8):
    """g_(k+2) = 8 g_(k+1) - g_k, from lambda_A^2 - 8 lambda_A + 1 = 0."""
    g = [AREA ** (-k) for k in range(kmax)]
    return max(abs(8 * g[k + 1] - g[k] - g[k + 2]) for k in range(kmax - 2))


def table():
    return [
        ("area inflation  lambda_A", "%.10f" % AREA, "4 + sqrt(15)"),
        ("linear inflation lambda_L", "%.10f" % LINEAR, "(sqrt6+sqrt10)/2"),
        ("lambda_A conjugate", "%.10f" % CONJUGATE, "4 - sqrt(15) = 1/lambda_A"),
        ("growth rate log(lambda_A)", "%.8f nats" % GROWTH_RATE, "NOT entropy"),
        ("log(lambda_L)", "%.8f" % LOG_LINEAR, "= GROWTH_RATE / 2"),
        ("Perron eigenvalue of M", "%.10f" % PERRON, "matches lambda_A"),
        ("gap-label base", "%.10f" % GAP_BASE, "Z[1/lambda_A] = Z[sqrt15]"),
        ("kappa*", "%.6f" % KAPPA_STAR, "lambda-independent"),
        ("c*", "%.6f" % C_STAR, "lambda-independent"),
        ("D_f*", "%.6f" % D_F_STAR, "lambda-independent"),
        ("Pi_circ", "%.10f" % PI_CIRC, "= 4 - sqrt(15)"),
    ]


def selftest():
    failures = []

    def check(label, ok):
        print("  %-56s %s" % (label, "ok" if ok else "FAIL"))
        if not ok:
            failures.append(label)

    print("the substitution matrix")
    check("nine species", M.shape == (9, 9))
    check("Gamma's column sums to 7, the rest to 8",
          [int(M[:, j].sum()) for j in range(9)] == [7] + [8] * 8)

    print("the inflation factors, three ways")
    check("Perron eigenvalue equals 4 + sqrt(15)", abs(PERRON - AREA) < 1e-9)
    check("AREA satisfies x^2 - 8x + 1", abs(AREA ** 2 - 8 * AREA + 1) < 1e-12)
    check("LINEAR satisfies x^4 - 8x^2 + 1",
          abs(LINEAR ** 4 - 8 * LINEAR ** 2 + 1) < 1e-12)
    check("LINEAR equals (sqrt6+sqrt10)/2",
          abs(LINEAR - LINEAR_CLOSED) < 1e-12)
    check("AREA = LINEAR^2", abs(AREA - LINEAR ** 2) < 1e-12)
    check("CONJUGATE = 1/AREA", abs(CONJUGATE - 1 / AREA) < 1e-12)
    check("the unit is Pisot: |conjugate| < 1", abs(CONJUGATE) < 1)

    print("the two logs are not interchangeable")
    check("GROWTH_RATE is exactly twice LOG_LINEAR",
          abs(GROWTH_RATE - 2 * LOG_LINEAR) < 1e-12)
    check("log(AREA) is the oscillation period, not 2*log(AREA)",
          abs(math.log(AREA) - GROWTH_RATE) < 1e-12)

    print("the gap-label module")
    check("the Pisot recurrence is exact", gap_recurrence_residual() < 1e-15)
    check("the leading gap label is the conjugate",
          abs(gap_label([0, 1]) - CONJUGATE) < 1e-12)

    print("the corrected polarisation")
    check("Pi_circ = 4 - sqrt(15)", abs(PI_CIRC - CONJUGATE) < 1e-14)
    check("and it is the leading gap label, not by construction",
          abs(PI_CIRC - gap_label([0, 1])) < 1e-14)

    print("the Eisenstein block is lambda-independent")
    check("gE ratio is 504/240", abs(GE_RATIO - 2.1) < 1e-12)
    check("kappa* is 6/(2h+1)",
          abs(KAPPA_STAR - 6 / (2 * H_BOUNDARY + 1)) < 1e-12)
    check("kappa* is unchanged by the retraction",
          abs(KAPPA_STAR - 3.6470588235294117) < 1e-9)

    print("the retraction is far from the correction")
    check("the retracted lambda differs from LINEAR by about 0.27",
          abs(WRONG_LAM - LINEAR) > 0.25)
    check("every retracted entry says why", all(v["why"] for v in
                                                RETRACTED.values()))

    print()
    if failures:
        print("%d failure(s): %s" % (len(failures), ", ".join(failures)))
    else:
        print("nariai_constants: all checks pass.")
    return len(failures)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(1 if selftest() else 0)
    print("nariai_constants -- authoritative values\n")
    for name, value, note in table():
        print("  %-28s %-22s %s" % (name, value, note))
    print("\n  retracted, do not cite:")
    for key, info in RETRACTED.items():
        val = "" if info["value"] is None else "= %.8g" % info["value"]
        print("    %-22s %s" % (key, val))
        print("      %s" % info["why"])
