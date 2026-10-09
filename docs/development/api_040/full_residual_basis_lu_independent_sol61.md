# Independent source/host reception of FullResidualBasisLU@1

Reviewed production: `6ea298945e6befc0a65143b5d51c506b7131f8b3`, parent
`6596a1749c98861ef87f4b469a39d94e428c40bc`, followed by budget encoding fix
`2a9e127da9eedaed9ba490c6c4c708f96a030f47` and scratch JSON projection fix
`777aa66584f0725193e2356f42f8cb742cd42195`. Independent checkout:
`/Users/romaindespoulain/dev/tmp/pops-sol61-full-lu-independent`.
The review adds only this report and two independent probe files.

## Finding and received correction

On exact `6ea29894`, a declared valid `max_dense_bytes=2**64-1` passed the public
Newton constructor and resource validator, then `prepare_spatial_newton` raised
`OverflowError: canonical CBOR integer at $.payload.resources.max_dense_bytes is
outside signed int64`. The independent first run had **12 PASS / 1 FAIL** at
this actual source seam. This was a contract defect, not a numerical failure.

Fix `2a9e127d` retains exact positive uint64 public/prepared integers and encodes
only the new budget domain as `{"uint64_hex":"ffffffffffffffff"}`. The common
legacy CBOR encoder is unchanged. Independent checks receive the upper bound,
identity distinction, strict 16-character lower-case hexadecimal decoding, and
refusal of bool/float/zero/out-of-range/extra-field/wrong-scope data. Fix
`777aa665` recursively projects this nested declaration for scratch reporting;
the independent real Case/resolve/emit and `ScratchPlan.to_json` controls receive
that export at uint64 max. No production blocker remains demonstrated by these
bounded probes.

## Evidence

The standalone C++ probe includes the actual production dense LU header. Debug
and NDEBUG each execute **385 assertions**: all 36 independent row/column
permutations of a nonsymmetric indefinite matrix with a zero initial pivot, a
strongly nonnormal triangular matrix, original matrix-vector residual checks,
stationary repeated applications, aliased RHS/output, revoked assembly readiness,
wrong shape, singularity, NaN/Inf, finite factorization overflow, and checked byte
arithmetic. These are ordinary host matrix controls, not fabricated native AMR
states or a Python numerical runtime.

The actual pinned `visit_active_cells_` and contribution predicate are compiled
and executed with two host rank adapters implementing reductions/error votes.
Six cases cover distributed ownership, an empty rank, replicated contribution
from lane rank zero, and collective refusals for duplicate owner, missing owner,
and nonfinite active mask. Covered/EB-inactive stored rows are represented by
zero entries of the provider mask and excluded from the quotient; every positive
adapter receives the same eight active component DOFs. This exercises the actual
scan against adapters. It does **not** execute a native EB geometry, AMR provider,
Kokkos kernel, or MPI communicator.

Two public source controls use the existing generic mixed nonlinear FieldProblem
helper with two/three unknowns and permutations. FullLU selection leaves the
complete source contract, local expressions, equation identity, and seven Newton
controls equal to the None realization, emits exact uint64 max, serializes scratch
resources, and chooses ProgramIR13. The default remains IR8. The current Uniform
provider refuses the selected AMR realization explicitly during resolution;
this is an implementation scope, not a permanent restriction on generic fields.

Frozen source seams establish that:

- The quotient requires exactly one contributing physical owner and matches the
  existing Krylov active threshold. It is distinct from the old stored-DOF Jacobi
  traversal, whose preparation body remains byte-identical to the parent.
- Overflow-checked matrix/map/vector/three-tower requirements are voted before
  the new FullLU allocations. Preparation snapshots the iterate and reuses the
  existing owning capture snapshots, lane, per-level attempts, topology,
  materialization generation, equation/point identities, and budget agreement.
- Every column invokes the actual full central JVP at that snapshot. That JVP
  calls original F twice, synchronizes the candidate before evaluating D(q), and
  includes the original local body. No diffusion-only diagonal substitutes for F.
- The factor rebuild is once per Newton iterate, outside GMRES. Every right
  application re-authenticates the quotient before gathering/solving. The actual
  J-delta path and terminal original F recheck remain present; the preconditioner
  does not replace either acceptance guard.

Four complete legacy source receipts were independently replayed against an
exact parent Git archive: implicit stage, nonlinear-map implicit stage, mixed
linear, and permuted mixed linear. Their IR and emitted C++ hashes match exactly:

| Control | IR SHA256 | C++ SHA256 |
|---|---|---|
| implicit-stage | `46bc2b8a4015801f7bfb9382decd793720370ffcc7c7967af974ab7e2d2b965f` | `a85a6229731a97d5d38a41a48ca64cff31093ec7e21d9b23423dc021fe1301b4` |
| nonlinear-map | `fbdbe9184a82270a0ce9081b00bae6579f9b0f53da082befa707eeb5f7f4748c` | `f375c2be1e389f3300d185b70a7bc2d1c9d592c131f13aa7d31b1df0931587d2` |
| mixed-linear | `71d46dd85de78e5aa8ca329a19374fb8baea3fc1693bff5b1dcd03da835e5a1d` | `e868f14782300a0c6dc14aadac8991603832218ac0cae9ce94a497b0701df780` |
| permuted | `5eca82535cea7bdc1dcc27487706f85aec4456d3f82bbf5929a9f0db544aca73` | `9a29dfffe91349f4ae35d3a24f8fc946b0d35cbea30e909fe318eaeb2baae928` |

## Commands, costs, and pending native scope

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python \
  PYTHONDONTWRITEBYTECODE=1 \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m pytest -q --tb=short \
  tests/review/test_sol61_full_lu_independent.py
```

The host tests invoke `c++ -std=c++20 -O0 -I<checkout>/include`, with NDEBUG for
the second dense run and `-pthread` for the active-scan adapters. No PoPS extension,
SDK installation, JIT, shared environment, or repository build runs. Ruff and
`git diff --check` are separate source checks. Legacy replay command:

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONDONTWRITEBYTECODE=1 \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B \
  tests/review/sol61_spatial_field_legacy_parity.py TREE
```

`TREE` is each of the current checkout and
`/Users/romaindespoulain/dev/tmp/sol61-full-lu-parity-20261001/parent-6596a174`,
the exact archive of the reviewed production parent. Final coherent receipt:
**18 PASS**, 22.93 seconds; Ruff and `git diff --check` pass. Earlier public-test
authoring mistakes (`_to_ir` instead of `_serialize`, and expecting the consumer
exception instead of the actual resolution ValueError) were corrected in the
probes. They were not production defects and are not included in the budget
finding above.

Costs are explicit: 2*Nactive original F evaluations per nonlinear factor
rebuild, replicated O(Nactive²) dense storage, O(Nactive³) partial-pivot LU,
and three extra numeric towers. The current quotient scan performs scalar
collectives per stored cell, including on every right application; it can be a
latency cost in MPI. The declared byte budget covers dense arrays/map/towers,
not wall time or every existing provider/Krylov allocation. No runtime cap or
physical/numerical tolerance is introduced by this review.

Root must receive the real AMR provider with partial coverage, actual EB/multiple
patches, distributed/replicated/empty ranks, one-rank budget/allocation/authority
failure, and changed quotient/capture/lease refusals. Real original nonlinear
F/D(q), strict terminal residual, iteration-budget behavior, candidate publication,
rollback/retry, and serial/MPI N32 results remain native obligations. This review
does not claim integrated coexistence with the separate Stage Q/tau IR12 bridge,
GPU/float32/3D qualification, or closure of the historical Identity GMRES failure.
