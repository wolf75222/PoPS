# Compensated sums for the explicit Program vector pairing

Date: 2026-09-30. Base: `f36176fa2a822c67b5ccc114bd21de1bef4be23e`.
Scope: only `pops.program.dot-all@1`; no change to endpoint policy, physical
method, target times, Python IR, historical dot/norm/component reductions, or
`dot_all_local` used by other algebra/solvers.

## Exact observed defect

The installed SDK506 receipt is
`outputs/installed-computed-dt-bound-public-dim2-sdk506-20260930/identity.json`
in the coordinator workspace. It authenticates source `f36176fa`, Dim2 extension
SHA256 `121bac0fd893427742e02a31437451a15f9ac7099274993c72d73c52fd7082f7`, and
SDK signature `506dce78009f1c0822b40ae1c38d9849eb9fbc2b65051906c4b2de4377039aaf`.
The installed `mf_arith.hpp`, `for_each.hpp`, and `program_context.hpp` hashes
match that receipt's `source-files.json`.

The two authentic saved states are `pytest-tmp/test_native_scalar_frontier_ro0`
and `...ro1`, each containing `computed_rotation.npz` under that output directory.
Their SHA256 values are respectively
`6f99f83b77ab0971be3c97134276ec52dd067e09df5ccd7b8433babc33d8247c` and
`9c1fef4ee223dd73a527926772993e9aaf0ba77fab2ee0ec0e708e8ccb582138`.
The saved field is constant on 64 cells: binary64 `(0.6, 0.8)` or
`(0x1.c3c3c3c3c3c3cp-1, 0x1.e1e1e1e1e1e1ep-2)`.

Replaying the actual Heun expression and the old accumulation order (64 cell
products summed separately for each component, then the two patch totals added)
reproduces the reported native times **exactly**:

| Requested second endpoint | Old sum endpoint | Error in endpoint ULPs | Exact sum of rounded products endpoint |
| --- | --- | --- | --- |
| 1.6 | 1.599999999999999 | -5 | 1.5999999999999999 (-1 ULP) |
| 0.9411764705882353 | 0.9411764705882342 | -10 | 0.9411764705882353 (0 ULP) |

The replay must use an explicit `sum += value` loop: Python 3.12's built-in
`sum(float)` already improves accumulation. A single naive loop over all 128
products has a different rounding history and does not reproduce this receipt.
No target is taken from a failed native result. `endpoint_ulps=4` stays unchanged.

## Implementation

`FiniteCompensatedSum` is a finite Neumaier two-word summary. A separate Kokkos
reduction functor accumulates each product and joins both words of partial
summaries. It shares the old exact product/mask kernel. `dot_owned_active_all`
keeps both words through patches and components; its existing Real-returning
entry point remains a wrapper. The explicit AMR provider keeps both words through
the already authenticated finest-owner levels.

The Uniform and AMR explicit providers pass the summary to
`collective_finite_compensated_sum`. The existing owner/layout/width/ghost checks,
replica validation and rank-zero replica contribution remain in place. Local
exceptions converge before transport. The new transport votes for finite local
summaries, votes for allocation success, allgathers **two Real scalars per rank**,
merges them in communicator-rank order, then votes for finite global output.
No field is gathered, and no rank-count ceiling is introduced. Each rank stores
2P scalars and performs O(P) merge work; MPI traffic grows with participants and
adds status collectives. This is an accuracy-oriented implementation, not the
scalability profile of one scalar MPI_SUM.

Keeping both words across MPI is necessary: rank0 `[1e16, 1]` and rank1 `[-1e16]`
have finite inputs/products/summaries, but separately finalizing local summaries
then applying scalar MPI_SUM produces zero. The new two-word merge retains one.
The C++ fixtures preserve that old counter-case and exercise the new real lane.

Every contributing input/product, mask, intermediate summary and final sum must
remain finite. Invalid flags propagate through joins. Product overflow, kernel,
patch/component/level accumulation overflow and global overflow still refuse
explicitly. A transient unrepresentable sum can still be refused even when a
later cancellation would make an exact mathematical sum finite; compensation
does not turn this into an arbitrary-precision accumulator.

The device reduction tree remains backend-dependent. Compensation improves
accuracy but does not promise correctly rounded sums, bit-identical GPU/CPU
results, invariance under arbitrary field partitions, or a universal four-ULP
bound for every problem. Rank-order merging prevents the demonstrated loss of a
local low word; it does not erase rounding inside local device trees.

## Verification

The exclusive checkout is `PoPS-sol61-dot-all-compensated`; the principal checkout
and installed environment are unchanged. The bounded standalone test links the
existing Kokkos CPU library against these source headers. Its real MultiFab,
FieldView, finite product kernel, custom Kokkos reducer and local helpers exercise
1D/2D, shifted origins, multiple patches, component/cell permutations, cancellation,
mask intersections, covered/inactive NaN, and strict overflow. The two rotations
use the saved binary64 states and unchanged endpoint bounds. This is local C++
host reception; it is not a public Program run on a rebuilt PoPS artifact.

Commands (one compiler process at a time):

```
rtk proxy clang++ -std=c++20 -O0 -Xpreprocessor -fopenmp -DPOPS_HAS_KOKKOS=1 \
  -Iinclude -I/Users/romaindespoulain/miniforge3/envs/pops-api040/include \
  tests/review/sol61_dot_all_compensated_kokkos.cpp \
  -L/Users/romaindespoulain/miniforge3/envs/pops-api040/lib -lkokkoscore -lomp \
  -Wl,-rpath,/Users/romaindespoulain/miniforge3/envs/pops-api040/lib \
  -o /tmp/sol61-dot-all-compensated-kokkos
rtk proxy env OMP_NUM_THREADS=1 OMP_PROC_BIND=false POPS_THREADS=1 \
  /tmp/sol61-dot-all-compensated-kokkos
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest \
  tests/review/test_dot_all_compensation_legacy.py \
  tests/python/unit/runtime/test_dot_all_native_contract_host.py -q
```

Real Kokkos CPU/serial-communicator reception: **52 assertions PASS**, with one
and two OpenMP threads. Source/host regression: **21 PASS**. Twenty byte-comparisons preserve both
providers' seven historical reductions, both AMR visitors, the finite product
kernel, historical `dot_all_local`, and both historical SUM overloads. The other
test runs authentic new local algebra and provider bodies with explicitly mocked
storage/collective transport. Its mock transport does not receive the new MPI
implementation.

The standalone test is also compiled with `-DPOPS_HAS_MPI=1 -lmpi` against real
MPI headers. The central C++ runtime fixture additionally exercises the real
duplicated lane, partitioned MultiFab with three patches per rank, cancellation
between rank summaries, empty summaries, poisoned last rank and global overflow.
That added fixture was separately syntax-compiled against real Kokkos/MPI/gtest
headers (PASS; one external gtest char8 conversion warning).
Multi-rank execution, rebuilt public rotation/restart runs, full AMR integration
and GPU execution remain pending central exact-SHA reception. No M27 or PDE
qualification is added. The unrelated public overflow fixture's exception-type
correction is owned separately by the independent reviewer.
