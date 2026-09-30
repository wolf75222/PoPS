# Explicit Python realization for the original AMR field residual

The public numerical choice is `Newton(right_preconditioner="SpatialBasisJacobi@1")`.
Define the physical `FieldProblem` equations first, register their `CellCenteredNonlinearCoupled`
method and this solver with the Case, then bind the exact State captures and evaluation point
in the Program. The Case field's `default_program_solver()` retains this choice.

```python
solver = Newton(
    tolerance=2e-9, max_iterations=12,
    linear_tolerance=1e-5, linear_max_iterations=240, restart=80,
    armijo=1e-4, minimum_step=1e-6,
    right_preconditioner="SpatialBasisJacobi@1",
)
# problem is the previously authored original physical FieldProblem.
field = case.field(problem, FieldDiscretization(
    method=CellCenteredNonlinearCoupled(finite_difference_step=1e-5),
    boundaries=(), solver=solver,
))
# Declare Program state/captures next, then:
request = field.bind_program_inputs(program, values=captures, at=point)
outcome = program.solve(request, solver=field.default_program_solver())
```

This selects the existing native `AmrFieldRightPreconditioner::kSpatialBasisJacobi` in
`PreparedAmrFieldResidual::prepare`; it does not evaluate a recognized model or replace
its source body. The emitter still builds the original local Kokkos body, full composite
operator, frozen captures, exact live hierarchy authority, one full-tower Newton solve,
SolveOutcome consumer and collective all-level candidate publication. The finite-difference
JVP remains that of the original full residual. Right preconditioning changes only the
linear correction realization; the original residual is recomputed before convergence.

The selected reference provider evaluates the **actual composite spatial operator** on zero
and each stored basis DOF, subtracts the zero response, and stores the reciprocal active-cell
diagonal. It adds one inverse field and costs **1 + stored spatial DOFs composite operator
applications at prepare**, including covered storage traversal. There is no size cap, assumed
spacing diagonal, SPD requirement, local-reaction derivative or equation-name dispatch.
Inactive covered entries remain zero. This cost is explicit and has no identity-default
side effect. A future efficient cached-diagonal provider needs its own authenticated
provider realization; it must implement the actual coarse/fine/reflux operator diagonal.
The reference realization does not claim such a provider exists.

`None` remains the default. Its numerical controls, solver identity, SolveRequest schema 1,
Program IR8, and emitted C++ remain unchanged. `options()` and `to_data()` include the named
choice only when selected; `numerical_options()` isolates the seven numerical controls.
The selected immutable prepared solver identity is `prepared-spatial-newton-v2`, its exact
right-preconditioner contract is `pops.amr.original-spatial-jacobi.basis-response@1`, its
SolveRequest is schema 2, and its Program IR is version 9. These fields enter request
validation and the compiled IR digest/source cache. Native invocation identity @2 also
includes the realization, current operator generation, owner, live attempt, captures,
point and prepared execution lane. There is no new native ABI, wire, or header manifest.

Uniform original-field residuals refuse this AMR-only provider during resolution, and the
Uniform emitter independently refuses it. Installed standalone nonlinear field plans and
other spatial implicit-stage adapters also refuse it explicitly before native work. These
are implementation limits of this provider; other mathematical residuals and later
providers are not restricted by this contract. Unknown, null-tagged or malformed named
realizations refuse admission even after a caller recomputes their request hashes.

## Source reception

Production source freeze: `c28a8f4370bbf2e9d781ddf52ddcc59597ce2e65` (eight Python files),
parent `3a03a2e` lane-consensus correction, atop native realization `c7cdd2c3`.
There are no C++ edits in the Python port.

Author tests exercise the existing independent physical witness at 2, 3 and 5 components,
permuted products, two captures, optional seeds and periodic/Neumann physical boundaries.
They validate/resolve/emit the original physical body and real AMR context route, test
policy mutations, Uniform/other-adapter refusals and detached/default-solver capture.
The exact original body-emission suffix is compared with c7. A separate identical Uniform
callsite against c7 and the new source preserves these hashes:

- Program IR `59ab71d259f6c2be65cfefab4b694dd3d92890b0a8ccc87f7e3ceef7db6db7df`.
- C++ `e3e9d0ce0ae356493526dd98bcb8888de2b17e12be21181c6c882f907890faa5`.
- The three physical module digests also match exactly in the companion receipt.

The frozen source suite receives 21/21 author checks in 108.56 s; the existing
mixed-field and implicit-stage tests receive 30/30 in 18.25 s. Ruff and diff-check
pass. The actual native core template probe instantiating dimensions 2 and 3
passes Clang syntax-only with the private headers and real Kokkos/MPI headers.

This is source authoring/resolution/emission evidence, **not native execution or public
AMR convergence qualification**. Root must rebuild the matching source/SDK and receive
serial/MPI execution, original saved residual, rollback/retry and checkpoint/restart.
The identity N32 iteration-limit evidence and preconditioned C++ profile remain distinct;
no claim is made that an unchanged identity realization converges within budget 240.
