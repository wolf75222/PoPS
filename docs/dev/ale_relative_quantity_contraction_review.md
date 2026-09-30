# Exact ALE quantity contraction counterexample

Author review base: `470dfb08774876aff411202c12491a02bf453197`, private
`PoPS-sol61-ale-receipt-analysis`. No source/environment/SDK/native artifact
was changed during the investigation. The AMR freeze `a8d6f04b` is preserved.

The source offset uses Fab's actual grown origin, axis stride and component
stride. Exact field marshaling independently enumerates the same component-major
grown storage. Negative/nonzero patch origins, ghost depths and permuted widths
do not produce an offset discrepancy. Independently decoded authentic saved
POPSEX03 images before the failing steps have correct source component rows and
accepted face quantities. Scalar source succeeds; noninteger vector source fails
at later steps, rather than at initial field/geometry projection.

Root's installed Dim1 diagnostic source is
`5f2076e3ff9ee8c1558022a3b0295bb39737d6f9`, SDK signature
`3f1286c0514edba336044d55c5e00b7965917455740adc31d0e21c621e30757d`,
native SHA256 `1cb02ada9cc8f0be2ed9dbc511becf77af37a243224f3d10cb88f0ae41995733`.
Its identity file SHA256 is
`8a6ae0c675b7f8cfb6e1a46c8d2d83f81fb07d48e3a6106dd6255f979729795e`;
log SHA256 is `664db77d1564d387b718dac928946294f9d350f95796930bc93476db7fcd88e5`.
Evidence directory: `outputs/installed-public-ale-exact-receipt-diagnostic-dim1-sdk3f12-20260930`.
These are root-executed native results; this reviewer performed only host
arithmetic and read-only evidence reception.

The exact accepted-exchange mismatches are:

| Case/step | Component/occurrence | Producer quantity | Receipt reconstruction |
|---|---|---|---|
| abc / 2 | 1 / cell:9/side:1 | -0x1.7ee75a8e3ebaap-11 | -0x1.7ee75a8e3ebabp-11 |
| cab / 3 | 2 / cell:1/side:1 | 0x1.fa10cfbb6fc96p-10 | 0x1.fa10cfbb6fc95p-10 |

For both occurrences, face measure and temporal weight are exactly binary64
`0x1p+0` on both sides. The quantity differences are independently reproduced
by exact rational arithmetic: ordinary two-rounding `physical - (density*sweep)`
versus single-rounding `fma(-density,sweep,physical)`. The operand triples were
reconstructed from the authentic saved previous states and declared projection;
the later native diagnostic confirmed the resulting quantity bits and exact
occurrences. No tolerance or guard was changed to obtain agreement.

This exposes a missing floating evaluation convention in POPSEX03. Its fields
store physical amount, face density and swept volume, but do not identify whether
the intermediate product was rounded separately. A portable new convention must
be named/versioned and applied by both producer and verifier. Supporting old
images must retain their old expression/bytes, rather than accept either nearby
quantity for a single declared convention. Root owns the production correction,
new wire04 convention and native reception. Independent codec04 injections will
follow its exact source freeze: noninteger three-component positive receipt,
unknown convention, exact quantity corruption, and historical03 byte equality.

`tests/review/test_sol61_ale_receipt_arithmetic.py` imports no PoPS/native module.
Seven host tests pass: two exact Fraction witnesses, externally pinned native
diagnostic matching, and four independent grown-storage offset enumerations.
Ruff passes. No compilation, JIT, environment installation or native execution
was performed here.
