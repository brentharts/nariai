"""argument_audit.py -- auditing the arguments, not the constants.

`consistency_test.py` checks that every script uses the right numbers and
that the identities between them hold. Passing it means the arithmetic is
sound. It says nothing about whether the *reasoning* downstream is sound,
and three scripts pass it while making claims their own code does not
support.

This module is the record of that audit. Each finding is a runnable
demonstration rather than a note, because a note in a README is lost at the
next rewrite and a failing check is not. Findings that are fixable have been
fixed in the scripts; findings that are structural are recorded here as
OPEN, with the demonstration that establishes them.

    python3 argument_audit.py
    python3 argument_audit.py --selftest    # the demonstrations still hold
"""

import math
import sys

import nariai_constants as C

FINDINGS = []


def finding(key, script, severity, claim, actual, status, note):
    FINDINGS.append(dict(key=key, script=script, severity=severity,
                         claim=claim, actual=actual, status=status,
                         note=note))


# ----------------------------------------------------------------------
# 1. brst_cohomology: the complex is a polygon, not the AP complex
# ----------------------------------------------------------------------

def polygon_betti(n):
    """Betti numbers of a single convex n-gon, i.e. a disk.

    This is what brst_cohomology.py actually builds. Its own docstring
    concedes the point -- "extra generators come from the substitution
    identifying edge-classes non-trivially" -- and then does not implement
    the identification. A disk has b = (1, 0, 0) for every n, so comparing
    n = 13 against n = 14 was guaranteed to return no difference before any
    tiling entered the question.
    """
    return (1, 0, 0)


def delta_chi_cohomological():
    return polygon_betti(14)[1] - polygon_betti(13)[1]


finding(
    "brst-disk", "brst_cohomology.py", "structural",
    "H*(Omega) of the Spectre tiling space, and a chirality cost "
    "Delta_chi = b1(Spectre) - b1(Hat)",
    "the homology of a single convex polygon, which is (1, 0, 0) for every "
    "n, so the difference is 0 for any pair of prototiles whatsoever",
    "OPEN",
    "The Anderson-Putnam complex is built by gluing prototiles along edge "
    "classes identified under the substitution. Without those "
    "identifications the object is a disk and carries none of the tiling's "
    "topology. Implementing them is a real piece of work, not a patch.")

finding(
    "brst-contradiction", "brst_cohomology.py", "high",
    "Delta_chi = 0 (printed from the table), and in the next sentence "
    "'b1 increases by Delta = 1 when going from Hat to Spectre'",
    "both statements printed together; the computed value is 0 and the "
    "asserted value is 1",
    "FIXED",
    "The script now prints the computed value and states plainly that it "
    "does not equal the edge-count Delta_chi used elsewhere.")

finding(
    "brst-torsion", "brst_cohomology.py", "high",
    "Hat has 2-torsion T = Z/2Z and the Spectre has none, offered as 'the "
    "homological signature of Delta_chi = 1'",
    "smith_normal_form_ranks returns torsion_count = 0 unconditionally and "
    "says so in its own comment: 'torsion requires exact SNF (skipped "
    "here)'. Nothing computes the torsion.",
    "FIXED",
    "Now labelled as an expectation from the literature rather than a "
    "result of this script.")


# ----------------------------------------------------------------------
# 2. The two incompatible definitions of Delta_chi
# ----------------------------------------------------------------------

def delta_chi_edges():
    """14 - 13, the definition nariai_constants uses and Pi_circ rests on."""
    return C.V_SPECTRE - C.V_HAT


finding(
    "delta-chi-ambiguous", "repository-wide", "high",
    "a single quantity Delta_chi, used as the numerator of "
    "Pi_circ = Delta_chi / lambda_A",
    "two incompatible definitions: an edge count (14 - 13 = 1) in "
    "nariai_constants, and a Betti-number difference (0) in "
    "brst_cohomology. The observational prediction is 12.70 percent under "
    "the first and exactly zero under the second.",
    "OPEN",
    "This is not a naming collision. Pi_circ is the repository's flagship "
    "prediction, and which definition is intended decides whether it "
    "predicts a signal or predicts none. The edge-count reading is the one "
    "in use; nothing argues for it over the cohomological one.")


# ----------------------------------------------------------------------
# 3. gap_label_spectrum: an irrationality test that fails on small numbers
# ----------------------------------------------------------------------

def best_rational(x, qmax=100):
    best = None
    for q in range(1, qmax):
        p = round(x * q)
        d = abs(x - p / q)
        if best is None or d < best[0]:
            best = (d, p, q)
    return best


def irrationality_absolute(k, qmax=100, tol=1e-12):
    """The original check: is lambda^-k close to some p/q in absolute terms?

    Unsound, and demonstrably so. As lambda^-k shrinks, the best rational
    with q < qmax becomes 0/1 and the absolute error becomes lambda^-k
    itself, which passes below any fixed tolerance eventually. The test
    therefore gets EASIER to fail the further you go, and declares the
    spectrum periodic for k >= 14.
    """
    g = C.AREA ** (-k)
    d, p, q = best_rational(g, qmax)
    return g, p, q, d, d < tol


def irrationality_relative(k, qmax=100, tol=1e-9):
    """Scale-free replacement: measure the error relative to the value.

    A rational p/q approximates x well in the sense that matters here when
    |x - p/q| / |x| is small, which cannot be achieved by sending x to zero.
    """
    g = C.AREA ** (-k)
    d, p, q = best_rational(g, qmax)
    return g, p, q, d / g, d / g < tol


def first_false_positive(kmax=40):
    for k in range(1, kmax):
        if irrationality_absolute(k)[4]:
            return k
    return None


finding(
    "gap-irrationality", "gap_label_spectrum.py", "high",
    "'All primary labels are irrational (Pisot property of lambda)', "
    "checked by requiring no label to sit within 1e-12 of a p/q with q < 100",
    "the check uses ABSOLUTE error, so once lambda^-k drops below the "
    "tolerance the nearest rational is 0/1 and the test reports the label "
    "as periodic. It first misfires at k = %s."
    % first_false_positive(),
    "FIXED",
    "Replaced by a relative-error criterion, which is scale free. The "
    "labels are of course irrational -- (4 - sqrt15)^k is irrational for "
    "every k -- so the conclusion was right and the evidence for it was not.")

finding(
    "gap-counting", "gap_label_spectrum.py", "medium",
    "26 aperiodic labels in (0,1) against 127 periodic ones, presented as a "
    "contrast between the two spectra",
    "both counts are artifacts of arbitrary truncation -- coefficients in "
    "{-1,0,1} with k <= 4 on one side, q <= 20 on the other. Z[1/lambda_A] "
    "is dense in R, so the aperiodic count grows without bound as either "
    "cutoff is relaxed.",
    "FIXED",
    "The counts are now labelled as truncation-dependent, with the density "
    "stated. The honest contrast is structural -- Z[sqrt15] versus Z[1/q] "
    "-- not a comparison of two finite tallies.")


# ----------------------------------------------------------------------
# 4. retrocausal: an identity presented as a verification
# ----------------------------------------------------------------------

def singlet_ratio_identity(r=None):
    """F_max/F_min - 1 with h := 1/(1+r). Returns (computed, r).

    nariai_constants defines H_BOUNDARY = 1/(1 + GE_RATIO). The script then
    forms F_max = 1/(1+h), F_min = h/(1+h), and reports that
    F_max/F_min - 1 equals GE_RATIO, asserting the match. But
    F_max/F_min = 1/h = 1 + r by construction, so the assertion is an
    algebraic identity in r and holds for every value of r. It cannot fail,
    and it therefore confirms nothing about singlet fractions, the Eisenstein
    coefficients, or the relation between them.
    """
    if r is None:
        r = C.GE_RATIO
    h = 1.0 / (1.0 + r)
    f_max, f_min = 1.0 / (1.0 + h), h / (1.0 + h)
    return f_max / f_min - 1.0, r


finding(
    "retro-circular", "retrocausal_capacity_verification.py", "high",
    "'r = F_max/F_min - 1 = 2.1 = |gE|/c_E4', asserted with "
    "`assert abs(r_from_singlets - gE_ratio) < 1e-12`",
    "an identity. h is DEFINED as 1/(1+r), so F_max/F_min = 1/h = 1+r for "
    "every r, and the assertion holds at r = 7, r = 1000, r = anything.",
    "FIXED",
    "Relabelled as the identity it is. The interpretation may still be "
    "worth making, but it is an interpretation and not a check.")

finding(
    "retro-retracted-objects", "retrocausal_capacity_verification.py",
    "high",
    "an operational meaning for the Eisenstein ratio in terms of I_max and "
    "I_doe, the max-information and Doeblin information",
    "corrected_constants.py retracts exactly those quantities: 'REMOVED: "
    "I_max, I_doe, C_retro as physical bounds -- category error; "
    "informational quantities for a different problem.' The script rests on "
    "objects the repository has withdrawn.",
    "OPEN",
    "Either the retraction is too broad or this script's premise is gone. "
    "The repository currently holds both positions. Resolving it is a "
    "judgement about the physics, not a code change, so it is left open "
    "and marked rather than quietly settled here.")


# ----------------------------------------------------------------------

def report():
    print("argument_audit -- what the constants checks cannot see\n")
    order = {"structural": 0, "high": 1, "medium": 2}
    for f in sorted(FINDINGS, key=lambda f: (f["status"] != "OPEN",
                                             order[f["severity"]])):
        print("  [%s] %-22s %s" % (f["status"], f["key"], f["script"]))
        print("      claimed : %s" % f["claim"])
        print("      actual  : %s" % f["actual"])
        print("      note    : %s" % f["note"])
        print()

    print("  demonstrations\n")
    print("  1. the AP complex that is a polygon")
    for n in (13, 14):
        print("       n=%d  betti = %s" % (n, polygon_betti(n)))
    print("     Delta_chi cohomological = %d, edge-count = %d"
          % (delta_chi_cohomological(), delta_chi_edges()))
    print("     Pi_circ would be %.8f or %.8f accordingly"
          % (delta_chi_edges() / C.AREA, delta_chi_cohomological() / C.AREA))

    print("\n  2. the irrationality check, absolute vs relative")
    print("     %-4s %-14s %-10s %-12s %-12s %s"
          % ("k", "lambda^-k", "best p/q", "abs err", "rel err", "absolute"))
    for k in (1, 4, 12, 14, 20):
        g, p, q, d, bad = irrationality_absolute(k)
        _, _, _, rel, _ = irrationality_relative(k)
        print("     %-4d %-14.4e %-10s %-12.2e %-12.2e %s"
              % (k, g, "%d/%d" % (p, q), d, rel,
                 "CALLS IT PERIODIC" if bad else "ok"))

    print("\n  3. the singlet identity, at values of r it was never meant for")
    print("     %-12s %-16s %s" % ("r", "F_max/F_min - 1", "match"))
    for r in (C.GE_RATIO, 1.0, 7.0, 1000.0):
        got, used = singlet_ratio_identity(r)
        print("     %-12.4f %-16.10f %s" % (used, got, abs(got - used) < 1e-9))
    print("     It matches at every r, so matching at 2.1 is not evidence.")

    n_open = sum(1 for f in FINDINGS if f["status"] == "OPEN")
    print("\n  %d findings, %d fixed, %d open."
          % (len(FINDINGS), len(FINDINGS) - n_open, n_open))


def selftest():
    failures = []

    def check(label, ok):
        print("  %-58s %s" % (label, "ok" if ok else "FAIL"))
        if not ok:
            failures.append(label)

    print("finding 1: the complex is a disk")
    check("a polygon has betti (1,0,0) regardless of n",
          polygon_betti(13) == polygon_betti(14) == (1, 0, 0))
    check("so the cohomological Delta_chi is 0",
          delta_chi_cohomological() == 0)

    print("finding 2: Delta_chi has two incompatible readings")
    check("the edge-count reading is 1", delta_chi_edges() == 1)
    check("they disagree", delta_chi_edges() != delta_chi_cohomological())
    check("and the prediction differs by everything",
          abs(delta_chi_edges() / C.AREA - C.CONJUGATE) < 1e-14
          and delta_chi_cohomological() / C.AREA == 0.0)

    print("finding 3: the irrationality check is not scale free")
    k0 = first_false_positive()
    check("the absolute test misfires at some finite k", k0 is not None)
    check("it passes at small k", not irrationality_absolute(1)[4])
    check("and fails at k=%s" % k0, irrationality_absolute(k0)[4])
    check("the relative test does not misfire anywhere tested",
          not any(irrationality_relative(k)[4] for k in range(1, 40)))

    print("finding 4: the singlet match is an identity")
    for r in (0.5, 2.1, 7.0, 1000.0):
        got, used = singlet_ratio_identity(r)
        check("holds at r = %g, so it cannot fail" % r,
              abs(got - used) < 1e-9)

    print("every finding names a script and a status")
    check("all findings have a script", all(f["script"] for f in FINDINGS))
    check("all findings are OPEN or FIXED",
          all(f["status"] in ("OPEN", "FIXED") for f in FINDINGS))
    check("the open ones are the structural and definitional ones",
          {f["key"] for f in FINDINGS if f["status"] == "OPEN"}
          == {"brst-disk", "delta-chi-ambiguous", "retro-retracted-objects"})

    print()
    if failures:
        print("%d failure(s): %s" % (len(failures), ", ".join(failures)))
    else:
        print("argument_audit: every demonstration still holds.")
    return len(failures)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(1 if selftest() else 0)
    report()
