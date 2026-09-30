# Original mixed spatial residual, first realization

`pops.spatial-field-residual@1` realizes an ordered product of independent
`FieldProblem` unknowns through the existing public `bind_program_inputs` →
`SolveRequest` → `Program.solve(...).consume(...)` → `field.observe` route.
The algorithm is selected by `CellCenteredNonlinearCoupled(finite_difference_step=...)`
and an explicit `Newton` descriptor. Neither a model name, component spelling,
fixed product dimension nor invertible field block selects this route.

The admitted equations have constant, finite, possibly nonsymmetric diffusion
and local nonlinear reactions/right-hand sides. For an ordered product `q`, the
native callback evaluates every original equation as

```
F_i(q; captures) = -sum_j D_ij Delta_h(q_j)
                  + local_i(q; captures) - rhs_i(q; captures).
```

The physical `Reaction`/`DivCoeffGrad`/`Laplacian` expressions determine signs
and coefficients. Scalar arithmetic, powers, square root and absolute value
are encoded as closed expression data, then authenticated again at emission.
Candidate reads address the exact ordered field handles. State reads retain
their exact qualified component, StateSpace, owner, temporal point and SSA
closure. Captures are copied into owned fields at invocation. They are separate
from the optional initialization seed; changing only that seed preserves the
equation identity and changes the initialization identity.

The source contract is immutable data. `ResolvedProgramFieldPlan` independently
recompiles the registered physical equations, ordered unknowns, selected method,
derivative step and boundary laws. Consequently, resealing both the residual
AST and its internal hashes cannot authenticate altered equations against the
registered `FieldProblem`. There is no live Case registry hidden in a detached
solve artifact.

Uniform Cartesian layouts with explicit periodic or homogeneous Neumann
relations are realized. Existing native boundary sessions and the general
matrix-free field stencil perform halo communication. The callback runs real
Kokkos kernels on real MultiFabs. The existing `PreparedSpatialResidual` owns the
candidate, central finite-difference work fields and persistent Newton/GMRES
workspace. Its selected derivative is

```
J(q) v = [F(q + s v) - F(q - s v)] / (2 s)
s = finite_difference_step * max(1, ||q||_2) / ||v||_2
```

The zero-direction derivative is zero. The finite-difference step is required,
positive and finite; it is encoded in the method and request identities. This
is the native full-residual derivative, including spatial coupling, rather than
a local reaction Jacobian or a Schur closure. Native Euclidean solver norms and
MPI lane reductions are used, without an invented volume weighting.

All captured fields and seeds receive an exact co-location/width preflight with
an MPI vote before copying/evaluating. A per-cell finite-status field and a lane
vote reject nonfinite original residual evaluations, including rank-local domain
errors, before entering the next collective solver stage. Native Newton/GMRES
also rejects nonfinite/overflowed norms and linear-algebra failures. After a
reported solve, the complete original residual is evaluated again. Its finite
L2 norm must satisfy the selected native stopping rule
`tolerance * max(1, reference_residual_norm)`. This actual norm replaces the
reported final norm and relative norm. Evaluation counts include this recheck.

Only then is the collective solve outcome consumed and the candidate copied to
its independent SSA output. Every invocation owns a different output even when
two solves share one request/template. Accepted States, histories, diagnostics,
time and macrostep remain under the existing Program transaction; failed solves
do not publish accepted data. No new runtime, dense global solve, inverse block,
condensation or custom publication transaction is introduced.

## Version and implementation ownership

New operation: `solve_spatial_field`. Public method:
`pops.fields.CellCenteredNonlinearCoupled`. Contract URI:
`pops.spatial-field-residual@1`. Emission includes the existing
`prepared_spatial_residual.hpp` and `general_field_operator.hpp`; this patch
changes no C++ header or native ABI/release contract.

The central integration must select ProgramIR **v8 only when this operation is
present**, including nested regions. The selector is
`python/pops/time/_program/serialization.py`, `_serialize`: its existing v5/v6
choice is followed by the recursive explicit-vector-pairing v7 scan. The v8
scan must take priority over v7 without stopping at the first `dot_all` node.
Release/SDK contracts and this conditional version selection are owned by the
central integration, not regenerated in this bounded author checkout.

Binder/authority: `python/pops/fields/_program_nonlinear_problem.py` and
`python/pops/codegen/program_field_plan.py`. Native body:
`python/pops/codegen/program_emit_nonlinear_field.py`. Solve protocol/barriers,
scratch accounting and observation checks admit the new operation explicitly.
The existing linear FieldProblem and `ImplicitStage` routes retain their data.

## Reception and limits

`test_nonlinear_mixed_field_runtime.py` is a public installed-package fixture,
reserved for central execution after rebuilding. It closes two/three original
equations on a 5×4 anisotropic periodic grid. Two distinct State captures come
from two genuine co-located blocks. Canonical and permuted unknown products are
solved twice with distinct initialization seeds. Exact Fourier cell means define
target DOFs; the forcing closes the declared discrete stencil and pointwise
reaction law. The fixture checks native observations against that target and
recomputes all original residual equations from actual saved native fields.
It saves NPZ and digest-linked receipts with native/module/artifact provenance,
actual time/macrostep and JUnit rank/dimension properties. One-iteration and
nonfinite-domain cases require collective refusal and unchanged accepted States,
histories, diagnostics, time and macrostep. These are fixtures, not results.

Local source checks exercise public resolution/emission, two/three unknowns,
permutations, two captures, seed separation, exact slot contracts, capture-clock
drift, wrong widths, invalid FD selections and resealed scientific AST forgery.
Whole emitted Dim2 C++ bodies pass syntax checks against actual headers. Four
separate-source comparatives against parent `2599d221` preserve **both** IR and
complete emitted C++ bytes for linear mixed normal/permuted cases and implicit
stages with/without a nonlinear physical map. The replay helper is
`tests/review/sol61_spatial_field_legacy_parity.py`.

No authentic solve, MPI execution, convergence campaign or GPU reception was run
in the author checkout. Central native reception and independent counter-review
remain required. AMR/hierarchy products, spatially varying or candidate-dependent
diffusion, gauges/nullspaces, extra preconditioners/outer solvers, auxiliary
provider captures and cross-layout/support transfer are refused in v1.
Per-unknown observations/history/field publication reuse the existing public
reconstruction route. The stricter existing linear `cell_mean_state` projection
contract is preserved; this tranche does not silently broaden field→State maps.

The handoff T3 and original corpus M27 describe original mixed equations and
forbid assuming universal Schur elimination. The supplied M27 registry closes
the stable **linear** periodic specialization; nonlinear double-well/wetting
variants remain planned and need their own physical closure, method and boundary
authority. Receiving this generic mechanism does not certify M27 nonlinear.

## Local commands and observed results

Source import is explicit; `PYTHONPATH` is unset. No environment setup/install or
JIT was performed. Interpreter:
`/Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python`.

```
env -u PYTHONPATH <python> -c 'import sys; sys.path.insert(0,"python"); import pytest; raise SystemExit(pytest.main([
"-q", "tests/python/unit/fields/test_nonlinear_mixed_field_problem.py",
"tests/python/unit/fields/test_m27_mixed_public_source.py",
"tests/python/unit/fields/test_m27_projection_independent_review.py",
"tests/python/unit/time/test_implicit_stage_request.py",
"tests/python/unit/time/test_implicit_diffusion_request.py",
"tests/python/unit/codegen/test_coupled_implicit_codegen.py",
"tests/python/architecture/test_import_graph.py"]))'
```

Result: **74 passed**, 42.05 s. The native fixture `--collect-only -q` receives
**5 tests collected**; none executed in this checkout. Ruff on all changed Python
files and `git diff --check` pass. A wider fields-directory lint pass reports six
pre-existing findings outside the patch (`_constant_mode_nullspace.py`,
`_field_matrix.py`, `problem.py`); these were not changed for this delivery.

The three-unknown, two-capture, two-invocation complete public fixture emits
`outputs/t3-spatial-nonlinear/native-fixture-program.cpp`. Its final syntax check:

```
/usr/bin/clang++ -std=c++20 -fsyntax-only -DPOPS_NATIVE_DIM=2 \
 -DPOPS_RUNTIME_SHARED_EXCEPTION_ABI -DPOPS_HAS_KOKKOS -DKOKKOS_DEPENDENCE \
 -DPOPS_HAS_MPI -DPOPS_HAS_PARALLEL_HDF5 -Iinclude \
 -I/Users/romaindespoulain/miniforge3/envs/pops-api040/include \
 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -x c++ \
 outputs/t3-spatial-nonlinear/native-fixture-program.cpp
```

The source-tree parity helper is run in separate Python processes against this
checkout and a `git archive 2599d221` export of `python`, the M27 example/imports
and `test_implicit_stage_request.py`. Both output JSON files compare exactly.
Digest prefixes (IR / complete C++):

| Authentic legacy case | Parent and candidate |
|---|---|
| Mixed linear | `71d46dd85de78e5a` / `e868f14782300a0c` |
| Mixed linear permuted | `5eca82535cea7bdc` / `9a29dfffe91349f4` |
| ImplicitStage | `46bc2b8a4015801f` / `a85a6229731a97d5` |
| ImplicitStage nonlinear mapping | `fbdbe9184a82270a` / `f375c2be1e389f33` |

These are source/host/structural checks, not CI or installed-native reception.
After checking, the generated bodies, parent export and complete parity JSON are
preserved outside the Git checkout in the workspace's
`outputs/sol61-t3-spatial-nonlinear-author-20260930/` directory. The commands above
record the paths used during the actual checks; relocation changes no payload.
