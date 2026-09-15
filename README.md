# nariai

Computational companion to

> **Aperiodic Vacuum Structure from BRST Cohomology on the Nariai Background**
> (B. S. Hartshorn, 2026)

and the related SLE, parity-violation and gravitational-wave analyses. The
repository holds the substitution-tiling algebra behind the paper, the
spectral and cohomological computations built on it, and a long series of
exploratory LIGO/pulsar data analyses.

Sibling repositories: [`spectre`](https://github.com/brentharts/spectre)
(the monotile substrate and its Lean fact layer) and
[`NariaiRelativeEntropy`](https://github.com/brentharts/NariaiRelativeEntropy)
(the relative-entropy derivation and Appendix B verification).

---

## Read this first: the repository contains a retraction

`corrected_constants.py` supersedes `verify_constants.py` and
`sle_constants.py`. The earlier two were built on an inflation factor

```
lambda = 2.53479630        <-- WRONG, root of 4x^4 - 8x^3 - 4x^2 - 4x + 1
```

which is not the Spectre inflation factor. The correct values, established
three independent ways in `corrected_constants.py` (Perron–Frobenius
eigenvalue of the substitution matrix, minimal polynomial, and the
closed form), are

```
area inflation    lambda_A = 4 + sqrt(15)      = 7.8729833462   (x^2 - 8x + 1)
linear inflation  lambda_L = (sqrt6+sqrt10)/2  = 2.8058837015   (x^4 - 8x^2 + 1)
```

The two numbers differ by 0.271, so **every λ-dependent quantity in the
superseded files is wrong**, not merely imprecise.

### Rebuilt — and there were eight, not five

Eight scripts computed with the retracted λ. All now import from
`nariai_constants.py`, the single importable source of truth, and
`consistency_test.py` fails if any of them drifts back.

| script | λ meant | rebuilt to |
|---|---|---|
| `gap_label_spectrum.py` | gap-label base | `GAP_BASE` (= λ_A) |
| `brst_cohomology.py` | gap-label base | `GAP_BASE` (= λ_A) |
| `parity_violation_derivation.py` | linear, with `LAM2` the area | `LINEAR` / `AREA` |
| `sgwb_polarization.py` | linear, with `LAM2` the area | `LINEAR` / `AREA` |
| `retrocausal_capacity_verification.py` | linear, with `LAM2` the area | `LINEAR` / `AREA` |
| `dimensional_flow_sle.py` | via `sle_constants` | `LINEAR` / `AREA` / `GROWTH_RATE` |
| `sle_phase_transition.py` | via `sle_constants` | `LINEAR` / `AREA` / `GROWTH_RATE` |
| `spectral_diagnostic.py` | Part 1 rebuilt entirely | see below |

Not a blanket substitution. `λ_A = λ_L²`, so a script using `LAM` for the
linear factor and `LAM2 = LAM**2` for the area stays internally consistent
when both move together. But the gap-label module is `Z[1/λ_A]`, so gap
labels are powers of the **area** factor — a script feeding the linear
factor into `sum(n_k λ**-k)` is wrong even with the retraction fixed.

**The three extra offenders were found by the test, not by inspection.** An
earlier draft of this README named five, from a grep for `2.5348` and for
`sle_constants` imports. `brst_cohomology.py`,
`retrocausal_capacity_verification.py` and `spectral_diagnostic.py` carried
the retracted value as an *un-evaluated expression* and never wrote the
digits anywhere, so that grep missed all three — and two of them were listed
here as authoritative. `consistency_test.py` searches for the form as well
as the decimal, which is the only reason they surfaced.

`spectral_diagnostic.py` was the worst case. Its Part 1, titled "Exact
Spectre invariants (certain)", derived a fourth-order gap-label recurrence,
verified it against the retracted λ, and reported the manuscript's appendix
recurrence as a bug by comparison. Both sides of that comparison are
obsolete: λ_A's minimal polynomial is quadratic, so the recurrence is
`g_(k+2) = 8 g_(k+1) - g_k`, second order. That block is replaced, not
annotated — a diagnostic that certifies a recurrence for the wrong number is
worse than one that certifies nothing.

`sle_constants.py` now **raises on import**. It is kept as a file so the
history stays legible, but a module whose only remaining purpose is to be
superseded should not quietly hand out a number.

### The second round: a correct constant in a broken identity

Fixing the constants was not enough, and the first rebuild was wrong in a
way that passed every check then in place.

Three scripts defined `H_TOP = log(LAM)` with `LAM` the **linear** factor,
then relied on `log(λ²) = 2·H_TOP` and `ν = H_TOP/π`. Rebuilding them by
mapping `H_TOP → GROWTH_RATE = log(λ_A)` kept the name and broke the
arithmetic: `log(λ_A)` is *already* `2·log(λ_L)`, so `2·H_TOP` became
`log(λ_A²)` and **the Pisot oscillation period silently doubled**, from
2.0634 to 4.1269. The crossover exponent ν doubled with it, 0.3284 → 0.6568.

Both numbers looked entirely plausible, and the source-, import- and
output-level checks all passed, because every constant involved was correct.
What was wrong was the relation between two of them.

| quantity | first rebuild | corrected |
|---|---|---|
| Pisot period `log λ²` | 4.12687414 | **2.06343707** |
| crossover exponent `ν` | 0.65681242 | **0.32840621** |

`nariai_constants` now exports `LOG_LINEAR` alongside `GROWTH_RATE`, with
the trap documented at the definition: they differ by exactly a factor of
two, which is precisely why substituting one for the other produces output
that looks right. `consistency_test.py` gained an **IDENTITY** stage that
checks relations between constants rather than the constants themselves, and
it has been verified against a deliberately reintroduced bug — restoring
`log_lam2 = 2*H_TOP` makes it fail, as it must.

`sle_phase_transition.py` was also still *printing* the retracted label
"Topological entropy h_top" next to the corrected number. The value moved in
the first rebuild; the name did not.

### The third round: auditing the arguments

Correct constants in correct identities still leave the question of whether
the reasoning downstream holds. Auditing the three remaining rebuilt scripts
line by line turned up seven findings, recorded as runnable demonstrations
in `argument_audit.py`. Four are fixed; **three are open, and two of those
bear on the flagship prediction.**

**Δχ has two incompatible definitions, and they disagree about whether there
is a signal at all.** `nariai_constants` defines `DELTA_CHI = V_SPECTRE −
V_HAT = 14 − 13 = 1`, an edge count, and `Pi_circ = Delta_chi / lambda_A`
rests on it. `brst_cohomology.py` defines it cohomologically, as
`b₁(Spectre) − b₁(Hat)`, and computes **0**. Under the first reading
Π_circ = 12.70%; under the second it is exactly zero. Nothing in the
repository argues for one over the other. *(open: `delta-chi-ambiguous`)*

**The AP complex is a polygon.** `brst_cohomology.py` builds a single convex
n-gon — a disk — whose Betti numbers are (1,0,0) for every n. Its own
docstring concedes the point: "extra generators come from the substitution
identifying edge-classes non-trivially", and the identification is not
implemented. So the Δχ = 0 above was guaranteed before any tiling entered
the question: any two prototiles give the same answer. The Anderson–Putnam
complex glues prototiles along edge classes identified under the
substitution, and building it is real work, not a patch.
*(open: `brst-disk`)*

**`retrocausal_capacity_verification.py` rests on objects the repository has
withdrawn.** `corrected_constants.py` retracts `I_max`, `I_doe` and
`C_retro` as physical bounds — "category error; informational quantities for
a different problem" — and that script's premise is their operational
meaning. Either the retraction is too broad or the premise is gone; the
repository currently holds both positions. *(open:
`retro-retracted-objects`)*

Fixed in this round:

* **An irrationality test that declared the spectrum periodic.**
  `gap_label_spectrum.py` checked that no gap label sat within `1e-12` of a
  `p/q` with `q < 100`, using *absolute* error. As λ⁻ᵏ shrinks the nearest
  such rational becomes `0/1` and the error becomes λ⁻ᵏ itself, so the test
  passes below any fixed tolerance — it first declares the labels periodic
  at **k = 14**. Replaced by a relative criterion. The conclusion was right
  — (4−√15)ᵏ = m + n√15 with n ≠ 0 — but the evidence for it was not, and a
  finite search over `q` cannot establish irrationality in any case.
* **A label count that measured its own truncation.** "26 aperiodic labels
  versus 127 periodic" compared coefficients in {−1,0,1} with k ≤ 4 against
  rationals with q ≤ 20. ℤ[√15] is dense in ℝ, so the aperiodic count grows
  without bound as either cutoff relaxes. Now labelled as such.
* **A circular verification.** `retrocausal_capacity_verification.py`
  asserted `r = F_max/F_min − 1 = 2.1 = |gE|/c_E4`. But `h` is *defined* as
  `1/(1+r)`, so `F_max/F_min = 1/h = 1+r` identically — the assertion holds
  at r = 7 and r = 1000 as readily as at 2.1, and cannot fail. Relabelled as
  the identity it is.
* **Torsion asserted while nothing computed it.** `brst_cohomology.py`
  reported `T = ℤ/2ℤ` for the Hat and `T = 0` for the Spectre as "the
  homological signature of Δχ = 1", while `smith_normal_form_ranks` returns
  a torsion count of 0 unconditionally and says so in its own comment. Now
  labelled an expectation, not a result.

### What changed, concretely

The flagship observational prediction is affected. `sgwb_polarization.py`
computed the SGWB circular polarisation as `Pi_circ = Delta_chi / lambda^2`
with `Delta_chi = 1`, giving **15.56%**. With the corrected area inflation
the same formula gives

```
Pi_circ = 1 / lambda_A = 1 / (4 + sqrt(15)) = 4 - sqrt(15) = 0.12701665...
```

that is **12.70%**, not 15.56%. The corrected value is not just a different
decimal: it is the conjugate unit of Z[sqrt 15], and it is exactly the
leading gap label `lambda_A^-1` in the authoritative summary table. Whether
the underlying identification is right is a separate question — but the
arithmetic downstream of it is now quoted at 12.70% throughout, and
`nariai_constants` checks that `PI_CIRC` equals the leading gap label rather
than assuming it.

Three other quantities are retracted outright in `corrected_constants.py`
and should not be cited from this repository at all:

* **topological entropy = 0.930 nats** — wrong twice over. It used the wrong
  λ, and primitive substitution tilings are uniquely ergodic with *zero*
  topological entropy. The well-defined quantity is the growth rate
  `log(lambda_A) = 2.0634` nats; label it as such.
* **`I_max`, `I_doe`, `C_retro` as physical bounds** — category error.
* **`r >= 0.01`** — a genuine result of Liu–Quintin–Afshordi quadratic
  gravity, imported rather than derived here. May be cited, not claimed.

---

## Authoritative numbers

From `python3 corrected_constants.py` (all assertions pass):

| quantity | value | note |
|---|---|---|
| area inflation `lambda_A` | `7.87298335` | `4 + sqrt(15)` |
| linear inflation `lambda_L` | `2.80588370` | `(sqrt6+sqrt10)/2` |
| `lambda_A` minimal polynomial | `x^2 - 8x + 1` | Pisot (quadratic) |
| `lambda_L` minimal polynomial | `x^4 - 8x^2 + 1` | Pisot (quartic) |
| `lambda_A` conjugate | `0.12701665` | `4 - sqrt(15) < 1` |
| inflation growth rate | `2.06343707` nats | `log(lambda_A)`, **not** entropy |
| gap-label module | `Z[sqrt 15]` | `= Z[1/lambda_A]` |
| gap-label recurrence | `g_(k+2) = 8 g_(k+1) - g_k` | 2nd order |
| spectral dimension `d_s` | `2.0 +/- 0.15` | measured |

The substitution matrix factors as `x^5 (x-1)(x+1)(x^2 - 8x + 1)`, and the
column sums are `[7, 8, 8, 8, 8, 8, 8, 8, 8]` — the 7 being the Mystic's
null slot. These agree exactly with the independent derivation in the
`spectre` repository, which is a useful cross-check: two codebases sharing
no source arrive at the same `4 + sqrt(15)`.

---

## Contents

### Core algebra (authoritative)

| file | purpose |
|---|---|
| `nariai_constants.py` | **Start here.** The single importable source of truth. Silent, derives everything from the substitution matrix, `--selftest`. |
| `consistency_test.py` | Checks the repository against itself: source, imports, output, identities between constants, and that the audited claims have not regressed. Run it before pushing. |
| `argument_audit.py` | The findings from auditing the *arguments*, each as a runnable demonstration. Three remain open. |
| `corrected_constants.py` | The retraction report: what was wrong, what replaced it, and why. |
| `spectre.py` | The Smith–Myers–Kaplan–Goodman-Strauss metatile substitution; the geometry everything else rests on. |
| `brst_cohomology.py` | Anderson–Putnam chain complex for the Spectre; BRST/boundary cohomology `H*(Omega)`. |
| `spectre_gaplabels_and_dimension.py` | Gap-label module from the Perron eigenvector, and the spectral dimension. |
| `spectre_spectral_dimension.py` | Spectral dimension of the tile-adjacency Laplacian on a genuine Spectre approximant, with finite-size calibration. |
| `spectral_diagnostic.py` | Measures what is measurable rather than assuming it; validated against Sierpinski (`d_s = 1.365` recovered). |

### Rebuilt on the corrected constants

| file | purpose |
|---|---|
| `gap_label_spectrum.py` | Gap-label spectrum vs a periodic baseline. |
| `sle_phase_transition.py` | Hat → Spectre transition as SLE_kappa; `kappa*` from the Eisenstein coupling `gE = -504`. |
| `dimensional_flow_sle.py` | Dimensional flow `d_s: 2 -> 4` via heat kernel and Pisot recursion. |
| `parity_violation_derivation.py` | Tiling-induced Chern–Simons coupling and the parity-odd VEV. |
| `sgwb_polarization.py` | SGWB circular polarisation prediction, now 12.70%. |
| `retrocausal_capacity_verification.py` | Ji–Lloyd–Wilde retrocausal capacity dictionary. |

The structure of these was never in question — the SLE and cohomology
arguments do not depend on λ's numerical value, only the final numbers did.

### Retracted, kept for the record

| file | status |
|---|---|
| `sle_constants.py` | Raises on import. Superseded by `nariai_constants`. |
| `verify_constants.py` | Superseded by `corrected_constants.py`. Historical. |

### Observational analyses (`ligo*.py`, `ligo_pulsar*.py`)

Thirty-six files — 19 analysis scripts and 17 LaTeX twins — forming an
exploratory series rather than a finished pipeline. `ligo.py` fetches strain data with local HDF5 caching; the
numbered successors iterate on aperiodic ratio tests, entropic coupling,
wavefront substructure, cross-detector correlation and attractor jitter.
`ligo_pulsar*.py` work on the PSR J1713+0747 profile-change event
([Zenodo 7236460](https://zenodo.org/records/7236460)). Each analysis script
has a `_latex` twin that emits the corresponding document, though the
pairing is incomplete: `ligo7`, `ligo11` and `ligo_pulsar` have no twin.

These have no README-level narrative yet and the numbering does not record
what superseded what. Treat them as a lab notebook.

---

## Running things

```sh
python3 nariai_constants.py             # the authoritative table
python3 nariai_constants.py --selftest  # verify every derivation
python3 consistency_test.py             # does the repo agree with itself?
python3 consistency_test.py --quick     # source and imports only, no runs

python3 corrected_constants.py          # the retraction report
python3 brst_cohomology.py              # AP complex and cohomology
python3 spectre_spectral_dimension.py   # spectral dimension measurement
```

Requirements: Python ≥ 3.10, `numpy`, `scipy`, `sympy`, `matplotlib`.
The `ligo*` scripts additionally need `gwpy` and `astropy` (and `sklearn`
for a few), plus network access to the GWOSC archive:

```sh
pip install gwpy astropy scikit-learn
```

### Current status

| group | files | runs clean |
|---|---|---|
| authority and test | 3 | 3 |
| rebuilt on the corrected constants | 8 | 8 |
| other core algebra | 3 | 3 |
| retracted, kept for the record | 2 | 1 (`sle_constants` raises by design) |
| `ligo*` / `ligo_pulsar*` | 36 | 2 without `gwpy` |

The 34 that fail do so only on the missing `gwpy` import, not on broken
code; they were not re-run against live GWOSC data in preparing this file,
so their status beyond "imports resolve" is unverified.

---

## Known problems, in the order worth fixing

1. **Resolve the three open findings in `argument_audit.py`** — see below.
   Two of them bear directly on whether the repository predicts a signal.
2. **Give the `ligo*` series an index.** Thirty-six files with no record of
   which supersede which is not recoverable by a reader, and barely by the
   author.
3. **Re-run the observational analyses.** Their last verified state is
   unknown; they were not run against live GWOSC data in preparing this file.
4. **Extend `consistency_test.py` to the `ligo*` series.** It currently
   checks the algebra only, because that is where the retraction was. If any
   of the observational scripts hard-code a tiling constant, the same drift
   can happen there and nothing would catch it.

## Papers

- https://doi.org/10.5281/zenodo.21303604
- https://doi.org/10.5281/zenodo.21382744
