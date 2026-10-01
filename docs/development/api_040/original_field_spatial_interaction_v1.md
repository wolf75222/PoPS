# Physical spatial interaction inside the original FieldProblem residual

This additive source extension implements `pops.spatial-field-residual@4`,
`pops.original-field-interactions@1` and
`pops.original-field-interaction-realization@1`. Program serialization selects
**IR20 only for this original-residual extension**. IR17/18/19 post-accept
interaction consumers remain separate. ROOT owns release/support declarations,
the rebuilt SDK and native reception.

The physical equation uses `pops.math.SpatialInteraction(unknown, kernel)` as an
inert, inspectable elliptic term. Its source is a declared FieldProblem unknown,
and its exact sign/scale, source component, row and kernel AST remain in the
physical equation identity. For example, `I - SpatialInteraction(rho, W) == g`
retains that original equation; rho and I stay simultaneous unknowns. The
mechanism never eliminates I or substitutes an accepted rho. Arbitrary ordered
unknown products, repeated terms, signed/nonsymmetric finite kernels and
unknown permutations use the same lowering. No model name or fixture dimension
is dispatched.

The numerical choice is explicit:

```python
quadrature = FieldInteractionQuadrature(
    CellVolumeMeasure(), CellMidpoint(),
    DirectSpatialInteraction(max_workspace_bytes=32 * 1024 * 1024))
method = CellCenteredNonlinearCoupled(
    finite_difference_step=1e-7, interaction=quadrature)
```

A physical interaction without this choice, or a choice without a physical
interaction, refuses. Serialized requests revalidate the equation correspondence;
code generation also recomputes it from the registered physical problem and
registered method. Descriptor rows alone cannot change registered physics.
Kernel/measure coordinate dimensions and declared units remain exact. The
current FieldProblem unknown Handle schema has no unknown physical-unit slot;
this extension does not invent a dimension for that absent declaration.

## Actual residual execution and authority

Uniform residual callbacks invoke the existing native direct-interaction kernel
on their actual q argument before consuming the detached result. AMR invokes a
collective producer inside `PreparedAmrFieldResidual::evaluate`, after the actual
FAC provider synchronizes the complete simultaneous q tower. This occurs for
every F, including both central finite-difference perturbations and the final
original-equation recheck. PerCandidate diffusion first evaluates the authored D
on that synchronized q; it uses the same original operator and F consumer.
Existing seven Newton/GMRES controls, residual sign, correction residual checks
and right-preconditioner choices remain intact.

The AMR Core issues a private lexical evaluation lease: exact candidate object
pointer, monotone uint64 nonce for every F invocation and provider association, alongside the existing
owner, attempt, operation points, topology/materialization/preparation epochs
and lane authority. A foreign tower with identical layout fails. The Context
checks that lease before transport and after detached construction. RAII revokes
it before the local F body and when the producer refuses. It cannot authenticate
an accepted or unperturbed `core.candidate()` alias as the current FD argument.
The nonce is not reset on retry; overflow refuses before issuing a new lease.
FD plus/minus on a reused q pointer cannot consume an old nonce.
No accepted state or public scratch map is published by the producer.

Native source cells use the actual prepared active, coverage, volume-fraction
and geometry getters. Existing direct interaction enforces unique ownership,
replica bit parity, finite source/kernel/output, empty-rank participation and
exact collective identity. Covered, zero-EB and ghost source cells do not
contribute. The current original AMR provider already refuses EB and asynchronous
original solves; this extension preserves those existing refusals and does not
claim original EB or subcycle qualification. The metric mechanism itself uses
native kappa times volume when such a provider is admitted.

The explicit direct workspace budget applies **per physical interaction term**:
transport/kernel workspace plus the retained detached AMR output tower, with
checked products/sums and a collective maximum of rank-local retained bytes.
The budget is checked before source transport and target allocation. Other
Newton/GMRES resources and other simultaneously retained terms are separate
resources. The interaction stores O(N) source/output data, never an N-by-N
matrix. An explicitly selected existing FullResidualBasisLU preconditioner has
its own matrix and separately chosen budget; it is not interaction storage.

## Bounded author checks (no native execution)

Private base: `a19462650cd01f76e0ac5f0a1611851b405a32a7`.

- `tests/review/sol61_original_interaction_source.py`: 6 SOURCE_ONLY unittest
  cases, actual Case validate/resolve/emission for Uniform and partial AMR,
  frozen and PerCandidate D, physical sign/component permutation, resealed
  physical-term mutations and explicit budget refusals. No `_pops` loaded.
- `tests/review/sol61_original_interaction_legacy_parity.py`: identical callsite
  run against a git archive of the parent and this source. Six noninteraction
  profiles (width 2/3, literal/captured/PerCandidate D) have byte-identical full
  IR, generated C++, discretization data and Module manifests. Joint JSON SHA:
  `71783e5de9b4abb431df5079b12eb936c4bfaf293e9081788aeaad032dc48ff0`.
- Eight actual import-graph architecture checks pass without pytest/native.
- One combined full TU includes all four generated public programs and the
  genuine complete `test_composite_general_field.cpp`, with private headers
  first and readonly NEWENV dependencies. AppleClang21, Dim2, MPI/Kokkos branch,
  `-fsyntax-only` exits 0. Two existing warnings remain (GTest char literal and
  ignored legacy SolveOutcome). TU SHA:
  `5ec3a37835196a90f88048ccd9f8440e93d3175d047c271389bc33df4a19d88e`.
  Private headers/TU hashes before/after are exact.

An earlier private syntax pass could not detect a producer callback that was
emitted but not invoked in the Core evaluation branch. This was found before
any gel/native claim and corrected. The native fixture now requires producer
calls equal actual residual evaluations, including perturbations, as well as
foreign same-layout and reused-pointer nonce rejection, plus lease revocation
after solve/refusal. The Core lexical scope revokes before local consumption.
Syntax alone cannot certify those execution assertions.

The native fixture is
`CompositeGeneralField.OriginalInteractionReevaluatesFullCandidateForEveryResidual`
in `tests/cpp/unit/elliptic/amr_original_field_interaction.inc`. It uses the actual
Newton/GMRES/FAC Core, nonconstant fields on a partial hierarchy, signed W,
permuted source components, original forcing and a final original residual.
It also verifies provider publication remains unchanged on solve/refusal and
budget/rank-local callback failure. It is compiled but **not executed** here.
ROOT must receive its serial/MPI run on the coherent rebuilt SDK before claiming
native success. No M26 continuum/PDE certificate, convergence, GPU or installed
end-to-end result is claimed by this source delivery.

Commands from this private checkout:

```sh
rtk proxy env PYTHONDONTWRITEBYTECODE=1 python3 tests/review/sol61_original_interaction_source.py
rtk proxy env PYTHONDONTWRITEBYTECODE=1 python3 tests/review/sol61_original_interaction_legacy_parity.py . outputs/original-field-nonlocal/legacy-candidate.json
# Run that unchanged exporter against git-archived parent-source as well; compare full bytes.
rtk proxy /usr/bin/clang++ -std=c++20 -arch arm64 -fsyntax-only \
  -DPOPS_NATIVE_DIM=2 -DPOPS_RUNTIME_SHARED_EXCEPTION_ABI \
  -DPOPS_HAS_KOKKOS -DKOKKOS_DEPENDENCE -DPOPS_HAS_MPI -DPOPS_HAS_PARALLEL_HDF5 \
  -I/Users/romaindespoulain/dev/tmp/pops-sol61-original-field-nonlocal/include \
  -isystem /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/include \
  -Xpreprocessor -fopenmp -I/Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/include \
  -I/Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops/work/PoPS/build-mpi/_deps/googletest-src/googletest/include \
  outputs/original-field-nonlocal/complete-generic-tu.cpp
```

Generated CPPs, logs and source-archive comparisons are preserved untracked in
`outputs/original-field-nonlocal/`; they are source/syntax evidence, not native
outputs or scientific saved-state seals.
