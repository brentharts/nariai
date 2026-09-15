"""sle_constants.py -- RETRACTED. Do not import.

This module exported the inflation factor

    LAM = (1 + sqrt3 + sqrt(2 + 2 sqrt3)) / 2

which is not the Spectre inflation factor, together with everything derived
from it: LAM2, H_TOP (itself a misnomer -- see below), and PI_CIRC.

It is kept as a file rather than deleted so that the history is legible and
so that `consistency_test.py` can name it, but it now raises on import. A
module whose only remaining purpose is to be superseded should not quietly
hand out a number: two scripts imported this one for months and printed
confident results, and nothing in either of them was wrong except this line.

Use `nariai_constants` instead:

    from nariai_constants import LINEAR as LAM, AREA as LAM2   # lengths/areas
    from nariai_constants import GAP_BASE                      # gap labels
    from nariai_constants import GROWTH_RATE                   # NOT H_TOP

The Eisenstein / SLE block that used to live here -- G_E, E4_COEFF,
GE_RATIO, H_BOUNDARY, KAPPA_STAR, C_STAR, D_F_STAR -- never depended on the
inflation factor and was never affected by the retraction. It has moved to
`nariai_constants` unchanged, values included: kappa* = 3.647059 before and
after.

`H_TOP` was called a topological entropy. Primitive substitution tilings are
uniquely ergodic and their translation action has topological entropy zero,
so the name was wrong independently of the number. The well-defined quantity
is the growth rate of the tile count, log(lambda_A) = 2.0634 nats.
"""

raise ImportError(
    "sle_constants is retracted: it exports the wrong Spectre inflation "
    "factor. Import nariai_constants instead "
    "(LINEAR/AREA for lengths and areas, GAP_BASE for gap labels, "
    "GROWTH_RATE in place of H_TOP)."
)
