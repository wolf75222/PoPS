# Independent reception of compensated dot-all, 9471bf8

Reviewed production SHA: `9471bf8ebceda549e75355ed0b9f2d77133d8a06`, parent
`f36176fa2a822c67b5ccc114bd21de1bef4be23e`. Reviewer: GPT-6.1 Sol, independent
of the author. No production changes, shared environment mutation, native build,
JIT or installed-package execution were performed for this reception.

## Result and evidence

No new production defect was demonstrated. Two independent tests and twenty
existing legacy byte-comparison tests pass (`22 passed`). The small C++ host
probe executes **105 strict assertions** using the actual frozen accumulator,
finite product/mask kernel and collective transport function bodies, extracted
with `git show` and checked against the current files. Only types, Kokkos
`isfinite`, compact field access, communicator votes and MPI transport are explicit
scaffolding. Consequently this checks actual arithmetic and transport protocol
logic; it does **not** execute Kokkos reduction trees, MPI processes or a native
Program.

The probe receives all six cancellation permutations, patch grains 1/2/3,
binary summary joins, empty rank summaries, and the counterexample
`[1e16, 1] | [-1e16]`. Its exact answer is 1 after transport of both words;
transport of finalized rank doubles would give 0. The send count/type/communicator
and the exact local high/low words are checked. Local and remotely voted invalid
summaries refuse before transport; allocation refusal is converged before
Allgather; global overflow or the post-merge finite vote refuses afterward.
The allocation case injects the allocation **vote**, rather than an actual OOM.
Inactive/covered NaN values contribute zero, malformed masks and active NaN or
overflowing products fail closed. Serial transport performs its finite vote
without Allgather.

The two saved SDK506 states and exact product sequences authenticated by
`sol61_dot_all_saved_state_oracle.py` are reused unchanged. At grains
1/2/7/32/64/128, actual compensated joins reach their requested endpoints within
the existing **four ULP** budget. Historical ordinary accumulation missed them
by 5/10 ULP; the independent exact sum of binary64 products gives -1/0 ULP.
This replay does not execute a new solver or change the saved states.

Source review verifies that the custom Kokkos reducer initializes the complete
summary and joins both words. The low part survives local patch/component joins
and AMR level joins. Both Uniform and AMR providers use the lane's actual
communicator and the new collective summary helper; they do not finalize then
perform a scalar MPI SUM. Replicated contributions are suppressed only after
local validation. Pointwise activity and finest-owner coverage stay in the
original finite product kernel. Historical `dot`, SUM and MAX reductions remain
unchanged; the twenty byte comparisons provide additional source evidence.

## Boundaries and risks

Allgather stores two Real values per rank, with O(P) memory and merge work on
every rank. Its allocation is voted collectively. It is neither a field gather
nor a claim of exact/reproducible arithmetic: device reduction trees and arbitrary
partition changes can still alter rounded summaries. Finite products whose
intermediate high/low accumulation overflows still refuse even when an exact
real sum would be finite; the probe preserves that contract. These checks do not
guarantee four ULP for every possible model.

Actual rebuilt CPU/Kokkos/MPI reception, the independent C++ cancellation fixture
from `31efe41617aa50ecf7cc0dda18c2ac7c7495b3c5`, and installed ComputedDt replay
remain the integrator's responsibility. Author-reported Kokkos host checks are
separate evidence and are not relabelled independent/native results here.

## Reproduction

From this checkout, using the specified Python executable:

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I -c 'import sys; from pathlib import Path; root=Path.cwd(); sys.path[:0]=[str(root/"python"),str(root)]; import pytest; raise SystemExit(pytest.main(["-q","--tb=short","tests/review/test_sol61_dot_all_compensated_independent.py","tests/review/test_dot_all_compensation_legacy.py"]))'
```

The frozen-file assertions intentionally fail on a different production header
revision. A later combined geometry reception must declare its new target rather
than treating this receipt as validation of changed headers.
