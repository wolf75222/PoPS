# Optional complete-residual basis LU, source contract

This change supplies `FullResidualBasisLU@1` as an explicit AMR numerical
realization of the original `FieldProblem`. It does not change the authored
equations, diffusion face formula, finite-difference step, nonlinear acceptance,
or any of the seven Newton/GMRES controls. It is neither a model dispatch nor a
default dense solver. Uniform has a named unsupported-realization refusal until
this particular provider is ported.

```python
solver = Newton(
    # the same seven numerical controls as the chosen physical problem
    right_preconditioner="FullResidualBasisLU@1",
    max_dense_bytes=64 * 1024 * 1024,
)
```

`max_dense_bytes` is an exact positive uint64 integer, excluding bool, float and
string. It is required only by this realization. Its version-1 resource
contract has scope `per_rank_dense_arrays_active_map_and_numeric_towers`.
Its canonical options/resources representation is a 16-digit lower-case `uint64_hex`
object, avoiding any change to the common signed-int64 CBOR domain. The authored
argument and retained prepared value remain exact integers. Raw float, bool, decimal
string or noncanonical hexadecimal resources are refused.
The budget participates in prepared solver identity, request realization,
Program IR13, emitted C++, scratch inspection and rank agreement. Numerical
controls remain a separate seven-key object. Deep-copying an authored solver
preserves the choice and budget; a prepared budget mutation fails its identity.

## Native operation and authority

At each Newton iterate q, the native full-tower workspace calls one explicit
preparation hook. The retained original provider supplies active masks and exact
patch distribution. Every stored cell must have one physical contributor on
the prepared lane; replicated storage contributes on lane rank zero, matching
the original provider's reduction. A covered coarse or inactive EB cell is not
a matrix row or column. Ordering is level, global patch, cell (axis zero fastest),
then ordered component. There is no component, model, resolution or DOF cap.

The resource owns a copy of q and an active basis/image tower. Each active basis
column invokes the same central JVP provider as Newton, hence two full original
F(q +/- h e_j) evaluations with the existing normalized finite-difference step.
This is a finite-difference Jacobian, not an analytic or SPD approximation.
For PerCandidate@1 each of those evaluations recalculates D at its own perturbed
q: synchronize/restrict q, evaluate D, restrict/halo D, composite flux, original
local terms. It includes the variation of D times the gradient of q. Captured
coefficients and physical local terms remain frozen at their exact declared
points. Captures are owned copies and the retained provider/attempt/level leases,
geometry generation, source identities and execution lane stay authenticated.

Actual active output values are gathered once per physical owner into a matrix
replicated on each lane rank. A native host partial-pivot LU factors it. Signed
and nonsymmetric matrices are allowed; an exactly singular full Jacobian,
nonfinite matrix/factors/solution, resource overflow or stale authority fails
collectively without publishing a candidate. This is a host numerical provider
inside the actual Kokkos/MPI Newton Program path, not a host replacement for the
Program or a solver run per AMR level.

The factor remains fixed within each GMRES solve of J_F delta = -F. The workspace
still applies the actual complete JVP to the complete correction and tests its
true residual; Armijo and the final original F recheck remain in place. The next
Newton iterate rebuilds the factor. Using SpatialBasisJacobi@1 with nonlinear
D remains explicitly incompatible; FullLU does not reinterpret that older
linear-spatial provider.

All possibly throwing local allocations, launches, numeric reads, factorization
and scatter phases fence and vote on the authenticated lane before the next
collective. Original F/JVP calls retain their own collective protocol. A foreign
lane argument is refused on the prepared lane before any foreign collective.
The active quotient is rechecked before a retained factor is applied.

## Cost and resource boundary

For N active components the dense resource consumes exactly, per rank:

- `sizeof(Real)*N*N` matrix bytes (8N squared in this SDK), in place LU;
- `2*sizeof(Real)*N` RHS and solution bytes;
- `sizeof(size_t)*N` pivot bytes;
- `sizeof(ActiveDof)*N` ordered active-index bytes;
- three local valid-cell numeric towers with the original unknown width.

Checked size_t products/additions and each rank's explicit byte limit are voted
before allocating these arrays. Rebuilding releases previous arrays before
replacement. This budget is not a process RSS limit: allocator overhead,
container objects, copied topology descriptors and the pre-existing original
provider/Newton/FD fields are explicitly outside its scope. Existing report
evaluation-counter encoding overflow is refused, never silently wrapped.

Preparation requires 2N original residual evaluations per Newton iterate,
O(N squared) matrix storage and O(N cubed) host factorization. The reference
active-map protocol performs two ordered stored-cell scans with owner/mask
votes; each right apply authenticates that quotient again. Matrix gathers use
native singleton numeric reductions, with O(N squared) such reads during
assembly. Each column and each right apply performs one globally reduced active
vector. These costs are disclosed; no hidden dense work enters Identity or
SpatialBasisJacobi. A future sparse/structured provider is a different explicit
realization rather than a change to this one's authority or physics.

## Source and pending native reception

The public source tests exercise scalar and coupled original FieldProblems,
captured D@2, D(candidate)@3, exact budget/refusal mutations, IR/emission and
scratch cost. A small unchanged numeric LU host-class probe checks multiple
pivots on a signed nonsymmetric matrix, singular/nonfinite refusal and checked
byte arithmetic. It has no AMR runtime, MPI execution or DSO.

The native fixtures retain the original equations, forcing, two resolutions,
permutations and seven controls. The existing positive original-field name now
explicitly selects FullResidualBasisLU@1 and records that realization. Identity
and Jacobi receive separate expected iteration-limit tests with invisible
candidates and unchanged published fields. Additional native fixtures test
budget and true singular-Jacobian refusal, and FullLU with varying D(candidate),
nonconstant three-component q, original recheck, failed acceptance/retry and
all-level publication. These fixtures are source/syntax evidence until Root
executes the rebuilt native target and MPI wrapper. No native convergence,
GPU qualification, public checkpoint receipt or M26/M27 campaign is claimed here.

Legacy Identity and SpatialBasisJacobi enum ordinals remain 0 and 1. FullLU is 2;
its prepared invocation is v4 while physical residual contracts @1/@2/@3 stay
unchanged. Only FullLU selects IR13 (the distinct EvolvedOriginalFieldStage extension selects IR12). Native ABI and checkpoint wire ordinals
are unchanged; the new actual SDK header changes the installed signature and
all native dimensions must rebuild. The public header is listed in the existing
header-manifest schema.


## Frozen author receipts

Production is the ordered chain `6ea298945e6befc0a65143b5d51c506b7131f8b3`,
`2a9e127da9eedaed9ba490c6c4c708f96a030f47`,
`777aa66584f0725193e2356f42f8cb742cd42195`, based on
`6596a1749c98861ef87f4b469a39d94e428c40bc`.
The first receipt remains a historical source defect: its accepted uint64 upper
bound was not expressible by the shared signed-int64 CBOR identity. The separate
second commit introduces canonical uint64 data. Its nested mapping export was
then closed in the third commit. Neither failure was fixed by relaxing a test or
by narrowing the admissible budget.

On frozen production 777: **25 source/host tests PASS in 133.38 s**; JUnit
`/tmp/sol61-full-residual-lu-closed-source.xml`. This runtime is a source-test
campaign cost, not a native LU benchmark. The unchanged numeric host LU probe
passed independently with multiple pivots on its signed nonsymmetric 3x3 data.
Ruff and `git diff --check` passed. Native cost remains unmeasured here; Root's
native fixture will record the actual active DOFs, dense bytes, factorization
and full-F evaluation counts for each resolution/permutation. Its JUnit runtime
will accompany those counts. No scalability claim follows from the host probe.

Six fresh historical images were exactly equal at the same retained fixture and
callsite: original Uniform IR8, captured Uniform IR10, original AMR Jacobi IR9,
captured AMR Jacobi IR10, candidate Uniform Identity IR11, candidate AMR Identity
IR11. Each image contains full IR/hash, emitted C++ hash, three Module hashes,
full manifests and complete request. The common image SHA256 is
`dc254843124ffc15378c267cde4cadcc1a50b2b86c157d52d2b42d19889687ec`.
Source-only receipt files are `/tmp/sol61-full-lu-parent-images.json` and
`/tmp/sol61-full-lu-current-images.json`, reproduced by
`/tmp/sol61_full_lu_parity.py` against fixed fixture directory
`/tmp/sol61-full-lu-parity-fixture`. The new canonical-budget corrections have
no executed path for those historical choices.

The real C++ syntax check passed with `POPS_NATIVE_DIM=2`, including the whole
`test_composite_general_field.cpp` and an explicit instantiation of
`PreparedAmrFieldResidual<2>::solve_candidate` with FullLU/PerCandidate. Two
pre-existing GTest/nodiscard warnings remain. The supplemental TU is
`/tmp/sol61_full_lu_final_syntax.cpp`. It instantiates production templates; it
has no run or native qualification.

Source replay (no `_pops` load/JIT/build/install):

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1   /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -c   'import sys; sys.path[:0]=["python","tests/review"]; import pytest; raise SystemExit(pytest.main(["-q","tests/review/test_sol61_full_residual_basis_lu.py"]))'
```

Root's native reception must reconfigure CTest discovery of the existing
`amr_original_field_residual.inc` fragment, compile with the new SDK signature,
and execute the original positive, the separate Identity/Jacobi negatives,
FullLU budget/singularity, FullLU candidate diffusion, all pre-existing tests,
and the real MPI aggregate wrapper. Public installed AMR and the future
TemporalTau/Q combination need separate source/native receipts after integration.
