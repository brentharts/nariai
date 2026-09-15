"""consistency_test.py -- does the whole repository agree with itself?

`corrected_constants.py` asserts its own values and always passed. That was
never the problem. The problem was that nothing asserted the *other* files
agreed with it, so five scripts went on computing with the retracted
inflation factor lambda = 2.5348 and printing confident output, for as long
as nobody happened to read them side by side.

This test is the missing assertion. It checks the repository against itself
in three ways, because each catches something the others miss:

  SOURCE   Every .py is scanned for the retracted constant and for imports
           of the retracted modules. This catches a file that has not been
           rebuilt, even if it is never run. Crucially it searches for the
           un-evaluated FORM as well as the decimal: three of the five
           offenders carried the value as
           `(1 + sqrt3 + sqrt(2 + 2*sqrt3))/2` and never wrote 2.5348
           anywhere, so a grep for the digits would have found none of them.

  IMPORT   Every rebuilt script must actually take its constants from
           `nariai_constants`. A file can be free of the retracted literal
           and still be wrong, by hard-coding a fresh copy of the right
           number that will drift the next time the right number changes.

  OUTPUT   The rebuilt scripts are run, and their stdout must contain the
           corrected values and must not contain the retracted ones. Source
           scanning cannot see a number assembled at runtime; this can.

  IDENTITY Relations BETWEEN the constants are checked, not just the
           constants. This stage exists because the first rebuild passed
           every other stage while being wrong. Three scripts defined
           `H_TOP = log(lambda)` with lambda the LINEAR factor and then
           relied on `log(lambda^2) = 2*H_TOP` and `nu = H_TOP/pi`. Mapping
           H_TOP to GROWTH_RATE = log(AREA) kept the name, satisfied every
           source and import check, and silently doubled the oscillation
           period, because log(AREA) is already twice log(LINEAR). A stale
           constant is easy to find; a broken identity between two correct
           constants is not, and it prints just as confidently.

`ALLOWED_TO_MENTION` is the exception list, and it is short on purpose. A
module that documents a retraction has to be able to name it -- but every
entry there is a file whose job is the retraction itself, and adding to that
list should feel like a decision rather than a fix.

    python3 consistency_test.py
    python3 consistency_test.py --quick     # skip the slow OUTPUT stage
"""

import os
import re
import subprocess
import sys

import nariai_constants as C

HERE = os.path.dirname(os.path.abspath(__file__))

# Files whose subject matter IS the retraction, so they must be able to name
# the retracted value without being flagged for it.
ALLOWED_TO_MENTION = {
    "nariai_constants.py",      # defines RETRACTED and the search patterns
    "corrected_constants.py",   # states the retraction and its reasons
    "consistency_test.py",      # this file
    "sle_constants.py",         # quarantined; see below
    "verify_constants.py",      # superseded, kept as a historical record
    "spectre_spectral_dimension.py",   # quotes the wrong value to correct it
}

# Scripts that must draw their constants from nariai_constants.
# Eight, not the five the README first named.  The three extra --
# brst_cohomology, retrocausal_capacity_verification and spectral_diagnostic
# -- carried the retracted value as an un-evaluated expression and were found
# only when this test searched for the form rather than the digits.
MUST_IMPORT = (
    "gap_label_spectrum.py",
    "parity_violation_derivation.py",
    "sgwb_polarization.py",
    "dimensional_flow_sle.py",
    "sle_phase_transition.py",
    "brst_cohomology.py",
    "retrocausal_capacity_verification.py",
    "spectral_diagnostic.py",
)

# Scripts whose output is checked. Kept to the rebuilt five plus the
# authority, because running the whole repository here would make the test
# slow enough that nobody runs it, which is how the drift happened.
OUTPUT_CHECKED = MUST_IMPORT + ("nariai_constants.py",)

# Strings that must never appear in the output of a rebuilt script.
FORBIDDEN_OUTPUT = ("2.5348", "2.53479", "0.93011", "15.56", "6.42519")


def python_files():
    return sorted(f for f in os.listdir(HERE) if f.endswith(".py"))


def scan_source():
    """Which files still carry the retracted constant, in any form."""
    hits = {}
    for name in python_files():
        if name in ALLOWED_TO_MENTION:
            continue
        text = open(os.path.join(HERE, name), encoding="utf-8").read()
        found = [pat for pat in C.RETRACTED_PATTERNS if pat in text]
        if found:
            hits[name] = found
    return hits


def scan_imports():
    """Which of the rebuilt scripts actually import the authority."""
    missing = []
    for name in MUST_IMPORT:
        text = open(os.path.join(HERE, name), encoding="utf-8").read()
        if not re.search(r"^\s*from\s+nariai_constants\s+import", text, re.M):
            missing.append(name)
    return missing


def run_one(name, timeout=300):
    proc = subprocess.run([sys.executable, name], cwd=HERE,
                          capture_output=True, text=True, timeout=timeout,
                          stdin=subprocess.DEVNULL)
    return proc.returncode, proc.stdout + proc.stderr


def scan_output():
    """Run the rebuilt scripts; their output must show the corrected values."""
    problems, seen = {}, {}
    for name in OUTPUT_CHECKED:
        try:
            rc, out = run_one(name)
        except subprocess.TimeoutExpired:
            problems[name] = ["timed out"]
            continue
        # nariai_constants prints the retracted values on purpose -- that is
        # what makes them searchable -- so the forbidden-output rule applies
        # to the rebuilt scripts only.
        bad = ([] if name == "nariai_constants.py"
               else [tok for tok in FORBIDDEN_OUTPUT if tok in out])
        if rc != 0:
            bad.append("exited %d" % rc)
        if bad:
            problems[name] = bad
        seen[name] = out
    return problems, seen


def main():
    quick = "--quick" in sys.argv
    failures = []

    def check(label, ok, detail=""):
        print("  %-54s %s%s" % (label, "ok" if ok else "FAIL",
                                "" if ok else "  " + detail))
        if not ok:
            failures.append(label)

    print("the authority agrees with itself")
    rc, _ = run_one("nariai_constants.py")
    check("nariai_constants imports and prints", rc == 0)
    check("its own selftest passes",
          subprocess.run([sys.executable, "nariai_constants.py", "--selftest"],
                         cwd=HERE, capture_output=True).returncode == 0)

    print("\nSOURCE: no file still carries the retracted constant")
    hits = scan_source()
    for name in sorted(hits):
        print("    %-38s %s" % (name, ", ".join(hits[name])))
    check("%d files scanned, none carry it"
          % (len(python_files()) - len(ALLOWED_TO_MENTION)), not hits,
          "%d file(s) do" % len(hits))
    check("the un-evaluated form is among the patterns searched",
          any("sqrt3" in p for p in C.RETRACTED_PATTERNS))

    print("\nIMPORT: the rebuilt scripts draw from the authority")
    missing = scan_imports()
    for name in missing:
        print("    %s does not import nariai_constants" % name)
    check("all %d rebuilt scripts import it" % len(MUST_IMPORT), not missing)

    if quick:
        print("\nOUTPUT: skipped (--quick)")
    else:
        print("\nOUTPUT: the rebuilt scripts print corrected values")
        problems, seen = scan_output()
        for name in sorted(problems):
            print("    %-38s %s" % (name, ", ".join(problems[name])))
        check("no forbidden value in any output", not problems)
        # and the corrected ones are positively present, not merely the old
        # ones absent -- a script that printed nothing would pass otherwise
        pol = seen.get("sgwb_polarization.py", "")
        check("sgwb_polarization prints the corrected polarisation",
              "0.1270" in pol or "12.70" in pol)
        gap = seen.get("gap_label_spectrum.py", "")
        check("gap_label_spectrum uses the area factor as its base",
              "0.1270166" in gap)
        check("and no longer calls the growth rate an entropy",
              "Topological entropy" not in gap)

    if not quick:
        print("\nIDENTITY: relations between the constants, not just their values")
        import math as _m
        check("GROWTH_RATE is twice LOG_LINEAR, so they cannot be swapped",
              abs(C.GROWTH_RATE - 2 * C.LOG_LINEAR) < 1e-12)
        flow = seen.get("dimensional_flow_sle.py", "")
        check("the Pisot oscillation period is log(lambda_A)",
              ("%.8f" % _m.log(C.AREA)) in flow)
        check("and is NOT twice it",
              ("%.8f" % (2 * _m.log(C.AREA))) not in flow)
        check("the crossover exponent uses log(lambda_L)",
              ("%.8f" % (C.LOG_LINEAR / _m.pi)) in flow)
        for name in ("sle_phase_transition.py", "dimensional_flow_sle.py",
                     "gap_label_spectrum.py"):
            check("%s does not call a growth rate an entropy" % name,
                  "Topological entropy" not in seen.get(name, ""))

    print("\nthe corrected values, for the record")
    print("    lambda_A  = %.10f   (4 + sqrt 15)" % C.AREA)
    print("    lambda_L  = %.10f   (sqrt6+sqrt10)/2" % C.LINEAR)
    print("    Pi_circ   = %.10f   (4 - sqrt 15)" % C.PI_CIRC)
    print("    kappa*    = %.6f       (unaffected by the retraction)"
          % C.KAPPA_STAR)

    print()
    if failures:
        print("%d failure(s): %s" % (len(failures), ", ".join(failures)))
        print("\nA failure here means the repository is publishing two")
        print("inconsistent sets of numbers again.  Rebuild the named file")
        print("on nariai_constants rather than adding it to ALLOWED_TO_MENTION.")
    else:
        print("consistency_test: the repository agrees with itself.")
    return len(failures)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
