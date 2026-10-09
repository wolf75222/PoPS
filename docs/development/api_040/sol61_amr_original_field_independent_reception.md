# Independent original AMR field capability review

Initial candidate: `104b0ffbba12a58f300aa00216f171249821775d`, parent
`b87b828095069b2a4b07ee2c96fac9ca715cdc78`. Point correction:
`b19e57945dbc9e2e95d8f424c52420fa7d38f88f`. This review owns only tests and
this report, in a separate checkout. Production fixes are authored separately.
The second received fix is `85a79b6f4c346f1a523470d38b14094fdf6226ef`.

## Counterexamples

The exact carrier constructor prefix was compiled with the genuine
`BoundaryEvaluationPoint` and `Rational` headers. Numerical allocation, provider
storage and MPI were excluded explicitly. On 104b0ffb, canonical images 0/1,
1/2 and 1/1 passed, but publicly mutated fractions 0/0, 2/1, 2/4 and -1/1 also
passed the point guard. The four expected refusal tests failed. All seven tests
pass unchanged on b19e5794. The new check precedes coefficient preparation;
the author's public native fixture remains pending native execution.

A second counterexample uses the unmodified Newton `dot_` and `local_phase_`
methods, with storage/kernel/MPI replaced by explicit trace substitutes.
The valid physical metric produces a preflight vote then a scalar reduction;
a local kernel failure votes and refuses before reduction. On b19e5794 a
left/right hierarchy-size mismatch instead throws before any vote (two refusal
ordering tests fail). This matters to the new route because its original-body
callback receives a mutable output tower: removing one level on rank0 without
throwing passes the body phase and the projection loop, then encounters the
unvoted size check before peers' collective. This is a demonstrated host control
flow defect and an MPI divergence risk, **not an executed MPI hang**. The author
has fixed it separately in 85a79b6f: output authentication occurs in the same
body vote; projection/dot/copy paths authenticate the entire tower inside their
guards. The same left/right size counter-tests now vote and refuse. Additional
width/layout mutations also vote and refuse before the sum. The native rank-local
body `pop_back` fixture has been added by the author but not executed here.

## Independent physics checks

The genuine `FluxMismatchTransfer` kernel is compiled unchanged, using explicit
host substitutions for indices and field storage. Eight cases cover both
interface sides, signed arithmetic coefficients, ratio2 in 1D, anisotropic
ratio(2,3) in 2D, and covered parents. An independent physical face-budget
calculation verifies `A = -div(D grad)`: replacing the coarse flux by the summed
fine flux makes coarse-volume and fine-volume interface contributions cancel.
Covered parents do not receive a second interface correction. These checks
receive the kernel arithmetic; they do not receive the prepared gather/halo,
patch-transfer routing or a complete native AMR application.

The actual general-provider `apply_impl_` method is also compiled unchanged.
Its scalar backend is explicitly substituted by the known modal action
`-Delta(q)=2.5q`, with a nonsymmetric signed three-component diffusion matrix
and a separate nonsymmetric reaction matrix. Two complete component
permutations match the independently computed `2.5 D q + R q`; transposing D
produces a different result. This receives row/column composition and selection
of arithmetic interpolation for full matrices, not a native differential solve.
Diagonal storage retains the existing harmonic interpolation realization;
full matrix storage uses arithmetic interpolation. No generic interpolation
equivalence or linear SPD authorization follows from apply-only support.

## Real source and syntax inspection

The capability is optional beside the unchanged base hierarchy solver interface.
The provider applies each original matrix entry through the existing composite
FAC operator and flux mismatch, then adds the authored reaction. It never
inverts the component matrix. The carrier retains the provider strongly, copies
capture values into owned arrays and uses one actual
`AmrFieldNewtonKrylovWorkspace` for the whole tower. Covered cells are excluded
from the metric, which uses actual level cell measures; replicated levels have
one physical contributor. The centered JVP evaluates the complete original
residual twice. A solved candidate is synchronized and checked against that
original residual again before becoming accessible.

`tests/review/sol61_amr_original_capability_syntax.cpp` independently instantiates
the genuine Dim2 provider, carrier and nonlinear Kokkos callback using actual
MultiFab interfaces. Clang `-fsyntax-only` passes with the real Kokkos/MPI,
OpenMP and parallel-HDF5 headers. No object was linked, installed or executed.
Local-body exceptions, copy/algebra votes and native transfer phase boundaries
were inspected; the explicit terminal unpack failure in existing
`PartitionedRegionTransfer` is not claimed retryable.

Final reception on **85a79b6f: 23 independent host tests pass** in 3.82 seconds;
the real Dim2 provider/carrier/Kokkos callback syntax also passes on this exact
revision. The tests reuse exact production methods and genuine point headers
where indicated; fake storage, scalar backend and collective traces are never
presented as native execution.

An additional **14 host authority tests pass** in 1.48 seconds. They compile the
exact `require_authority` and local guard with real `PreparedResourceCache`,
`PreparedResourceAttempt`, `BoundaryEvaluationPoint`, Newton options and exact
contract builder/serial lane. The operator metadata getter and Kokkos fence are
explicit substitutes. Foreign owner, foreign parent/per-level leases with the
same ordinal, original source/capture identities, widths, points, topology and
materialization epochs, coefficient generation, rejected parent/level and fresh
retry are refused. Invalid probes preserve the live baseline before revocation.
These guards do not execute capture allocation, a complete numerical carrier,
MPI consensus, native operator application or actual rollback publication.

## Reproduction

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 <python> -m pytest -q \
  tests/review/test_sol61_amr_original_point.py \
  tests/review/test_sol61_amr_original_flux.py \
  tests/review/test_sol61_amr_original_matrix.py \
  tests/review/test_sol61_amr_original_authority.py \
  tests/review/test_sol61_amr_original_collective.py \
  -p no:cacheprovider --disable-warnings
```

The tests compile small host programs with `/usr/bin/clang++ -std=c++20`;
their substitutions and excluded mechanisms are stated in each file. No PoPS
package import, numerical prototype, dense solve, JIT or shared environment
mutation is used. Ruff checks the independent Python probes.
The final combined command above receives **37 passed in 4.76 seconds** on
85a79b6f; Ruff and the staged diff whitespace check pass.

## Public connection remains separate

At b19e5794 the real Python emitter still rejects `target != "system"` for
`solve_spatial_field`, with the Uniform requirement. The current AMR barrier
enum has only linear solve, field publication and spatial RHS. Consequently
the capability alone **does not realize public T3 AMR FieldProblem**. The
next review must receive the actual original-field IR/scheduler connector,
genuine parent/per-level attempt leases, exact captures/seeds, SolveOutcome
consumption, observations and atomic all-level publication.

The existing independent public case
`test_sol61_spatial_field_cda_independent.py::test_uniform_only_realization_refuses_amr_target`
is replayed against this checkout's authentic Python source: **1 passed**, 3.66
seconds. It constructs and emits a genuine FieldProblem and confirms the AMR
target refusal. This refusal is preserved as a current limitation, not treated
as acceptance of the future connector.

The current carrier v1 explicitly requires synchronized time, duration and
stage fraction across levels. It does not establish subcycled-point support;
a connector must preserve genuine level points or refuse outside that policy,
without relabeling clocks. No M27 nonlinear boundary/closure, M14 science,
continuum convergence, native solve, MPI/GPU, restart or runtime retry is
qualified by these source/host/syntax checks. A forcing produced by the same
native operator is a closed discrete numerical witness, not an independent
certificate of physical stencil correctness.
