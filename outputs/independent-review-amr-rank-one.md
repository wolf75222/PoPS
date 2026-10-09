# Independent source review: AMR rank-one invalid-face witness

Reviewed `cd640d8` and the follow-up `5e6b0e6` in MAIN read-only. No native MPI
execution, timeout experiment, or installed-package qualification was performed
by this reviewer.

The local-box API reads the live level-zero MultiFab's rank-local boxes. The
Python runtime view exposes validated immutable half-open bounds. The witness
gathers both ranks' real boxes, checks exact non-overlapping coverage and total
counts, and selects a strict interior cell belonging to rank one and no rank
zero box. This is actual ownership evidence rather than a partition heuristic.
With the authored slope zero, its invalid left-state numerical-face branch is
triggered from that owned interior cell.

The original witness could pass on an unrelated RuntimeError and contained
rank-local assertions before collectives. These findings were sent to the
author. Follow-up `5e6b0e6` adds a same-artifact/layout/parameters positive run
with the invalid branch inactive, collective `_agree` assertions, and exception
capture before gathering error descriptions. The negative run checks unchanged
time/step and gathers both block states before the root compares arrays exactly.

Two remaining test limitations were confirmed with the author:

- `_world()` is called before explicitly selecting native dimension two;
  the fixture's compile helper selects it only later. An unselected native
  backend can therefore fail before exercising the intended witness.
- The error check still accepts any RuntimeError rather than identifying the
  numerical-face failure. The positive control substantially narrows ambiguity,
  but a specific numerical diagnostic should complete the proof.

An external MPI-process timeout remains necessary: collective assertion helpers
cannot rescue a rank stuck inside a native collective. The final root-only
array assertions occur after the state gathers; they do not themselves precede
a further collective in this test. Root retains responsibility for actual MPI2
execution, timeout enforcement, and installed artifact authentication.

No production defect was established in the ownership API itself by this
bounded source review. This one-level distributed witness does not qualify
multi-level refinement, arbitrary rank counts, or GPU failure behavior.
