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

### This is not contained — five live scripts still use the retracted value

The retraction is documented inside `corrected_constants.py`, but nothing
stops the other scripts from running and printing confident numbers derived
from the old λ. As of this writing:

| script | how the retracted λ enters | status |
|---|---|---|
| `gap_label_spectrum.py` | hard-coded `LAM = (1+sqrt3+sqrt(2+2 sqrt3))/2` | **needs rebuild** |
| `parity_violation_derivation.py` | same hard-coded line | **needs rebuild** |
| `sgwb_polarization.py` | same hard-coded line | **needs rebuild** |
| `dimensional_flow_sle.py` | `from sle_constants import LAM, LAM2, H_TOP, ...` | **needs rebuild** |
| `sle_phase_transition.py` | same import | **needs rebuild** |

All five run without error and produce plausible-looking output. There is
currently no way for a reader to tell them apart from the authoritative
results, which is the single most important thing to fix in this repository.

### What changes, concretely

The flagship observational prediction is affected. `sgwb_polarization.py`
computes the SGWB circular polarisation as `Pi_circ = Delta_chi / lambda^2`
with `Delta_chi = 1`, giving **15.56%**. With the corrected area inflation
the same formula gives

```
Pi_circ = 1 / lambda_A = 1 / (4 + sqrt(15)) = 4 - sqrt(15) = 0.12701665...
```

that is **12.70%**, not 15.56%. The corrected value is not just a different
decimal: it is the conjugate unit of Z[sqrt 15], and it is exactly the
leading gap label `lambda_A^-1` in the authoritative summary table. Whether
the underlying identification is right is a separate question — but the
arithmetic downstream of it should be quoted at 12.70%.

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
| `corrected_constants.py` | **Start here.** Authoritative constants, with every retracted quantity named and its reason given. |
| `spectre.py` | The Smith–Myers–Kaplan–Goodman-Strauss metatile substitution; the geometry everything else rests on. |
| `brst_cohomology.py` | Anderson–Putnam chain complex for the Spectre; BRST/boundary cohomology `H*(Omega)`. |
| `spectre_gaplabels_and_dimension.py` | Gap-label module from the Perron eigenvector, and the spectral dimension. |
| `spectre_spectral_dimension.py` | Spectral dimension of the tile-adjacency Laplacian on a genuine Spectre approximant, with finite-size calibration. |
| `spectral_diagnostic.py` | Measures what is measurable rather than assuming it; validated against Sierpinski (`d_s = 1.365` recovered). |

### Built on the retracted constants (do not cite)

| file | purpose |
|---|---|
| `verify_constants.py` | Superseded by `corrected_constants.py`. |
| `sle_constants.py` | Superseded; still exports `LAM = 2.5348`, `H_TOP = 0.930`. |
| `gap_label_spectrum.py` | Gap-label spectrum vs a periodic baseline. |
| `sle_phase_transition.py` | Hat → Spectre transition as SLE_kappa; `kappa*` from the Eisenstein coupling `gE = -504`. |
| `dimensional_flow_sle.py` | Dimensional flow `d_s: 2 -> 4` via heat kernel and Pisot recursion. |
| `parity_violation_derivation.py` | Tiling-induced Chern–Simons coupling and the parity-odd VEV. |
| `sgwb_polarization.py` | SGWB circular polarisation prediction. |
| `retrocausal_capacity_verification.py` | Ji–Lloyd–Wilde retrocausal capacity dictionary. |

Several of these are interesting independently of the constant they use —
the SLE and cohomology structure does not depend on λ's numerical value,
only the final numbers do. They are listed here because their *output* is
currently untrustworthy, not because the ideas are.

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
python3 corrected_constants.py        # authoritative constants, asserts throughout
python3 brst_cohomology.py            # AP complex and cohomology
python3 spectre_spectral_dimension.py # spectral dimension measurement
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
| core algebra | 6 | 6 |
| built on retracted constants | 8 | 8 (output untrustworthy) |
| `ligo*` / `ligo_pulsar*` | 36 | 2 without `gwpy` |

The 34 that fail do so only on the missing `gwpy` import, not on broken
code; they were not re-run against live GWOSC data in preparing this file,
so their status beyond "imports resolve" is unverified.

---

## Known problems, in the order worth fixing

1. **Rebuild the five live scripts on `corrected_constants`.** Replace the
   hard-coded `LAM` lines and the `sle_constants` imports with the verified
   values, and re-derive `kappa*`, `Pi_circ`, and the dimensional flow.
   Until then the repository publishes two mutually inconsistent sets of
   numbers with nothing marking which is which.
2. **Delete or quarantine `sle_constants.py`.** It is a live import path to
   a retracted value. A module that exists only to be superseded should
   raise on import, not return a number.
3. **Make the retraction machine-checkable.** `corrected_constants.py`
   asserts its own values; nothing asserts that the *other* files agree with
   it. A cross-file consistency test that fails when any script computes
   with `2.5348` would have caught all five.
4. **Give the `ligo*` series an index.** Thirty-six files with no record of
   which supersede which is not recoverable by a reader, and barely by the
   author.
5. **Re-run the observational analyses.** Their last verified state is
   unknown.

## Papers

- https://doi.org/10.5281/zenodo.21303604
- https://doi.org/10.5281/zenodo.21382744
