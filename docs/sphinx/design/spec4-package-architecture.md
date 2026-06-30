# Spec 4: Python package architecture

This page describes the target layout of the `pops` Python package introduced in
Spec 4 and tightened by the later clean-break specs. Python authors typed
objects, lowers them to `pops.model.Module` plus `pops.time.Program`, and
`pops.compile_problem(...)` emits the combined C++ artifact.

```{note}
No backward-compatibility shims are shipped for removed public routes. The
documented route is `compile_problem(...)`, then `System.install(...)`, then
`System.step_cfl(...)`.
```

## Target sub-packages

| Package | Responsibility |
|---------|---------------|
| `pops.ir` | Symbolic IR: expression nodes, ops, values, lowering passes, and visitors. Imports nothing inside `pops`. Used by every layer above it. |
| `pops.model` | Operator-first typed model core: `Module`, typed spaces (`StateSpace`, `FieldSpace`, `RateSpace`, `ParameterSpace`, `AuxSpace`), `Operator`, `OperatorRegistry`, `Signature`. Imports only `pops.ir`. |
| `pops.physics` | Math and physics authoring facade. `pops.physics.Model` is the high-level PDE description (conservative variables, flux, eigenvalues, sources, elliptic right-hand side). Lowers to a `pops.model.Module`. Imports `pops.ir` and `pops.model`. |
| `pops.time` | Temporal language: `Program`, schedules, equations, and the operator-first time IR. Imports `pops.ir` and `pops.model`. |
| `pops.numerics` | Finite-volume discretisation descriptors: spatial method, Riemann solver, reconstruction, limiters and numerical terms. |
| `pops.fields` | Field and elliptic problem descriptors (`PoissonProblem`, boundary conditions, outputs). |
| `pops.linalg` | Linear algebra objects: `LinearProblem`, `MatrixFreeOperator`, norms and reductions. |
| `pops.solvers` | Compiled solver descriptors: elliptic, Krylov, Schur and nonlinear routes. |
| `pops.mesh` | Mesh, layout, AMR, geometry, boundary and mask descriptors. |
| `pops.params`, `pops.diagnostics`, `pops.output`, `pops.external` | Typed runtime parameters, diagnostics, output/checkpoint policy and external compiled brick descriptors. |
| `pops.moments` | Generic moment-model authoring tools. Ready-to-use moment models live in `pops.lib.models.moments`. |
| `pops.lib` | Provided presets and ready models only. Generic construction tools live in their top-level package. |
| `pops.codegen` | The only C++ emitter. It consumes `Module`, `Program`, descriptors, layout and backend objects. Does not import `pops.runtime`, and never imports `_pops` at module scope. |
| `pops.runtime` | Thin facade over the `_pops` native extension: `System`, `AmrSystem`, and their configuration objects. Imports only `_pops`. |

## Acyclic import graph

The graph has a single direction: lower packages never import upper ones.
`pops.codegen` is the exclusive C++ emission point; authoring packages
(`pops.physics`, `pops.time`, `pops.lib`) never import it or `_pops`.

| Package | May import (inside pops) |
|---------|--------------------------|
| `pops.ir` | _(nothing)_ |
| `pops.model` | `pops.ir` |
| `pops.physics` | `pops.ir`, `pops.model` |
| `pops.time` | `pops.ir`, `pops.model` |
| `pops.numerics`, `pops.fields`, `pops.linalg`, `pops.solvers`, `pops.mesh`, `pops.params`, `pops.diagnostics`, `pops.output`, `pops.external`, `pops.moments` | `pops.descriptors` plus lower pure authoring layers as needed |
| `pops.lib` | ready-model/preset packages only |
| `pops.codegen` | `pops.ir`, `pops.model`, `pops.time`, `pops.physics`, descriptors and presets |
| `pops.runtime` | `_pops` only |

## Codegen as free functions: the key design decision

The compile and emission helpers live in `pops.codegen` as ordinary free
functions that accept typed model/program objects. They do NOT live on
`pops.physics.Model` as methods.

Why: keeping C++ emission out of the authoring packages prevents a cycle.
As free functions, `pops.codegen` may import everything above it while authoring
packages import nothing from `pops.codegen`. Callers that need to compile call
`pops.compile_problem(model=module, program=program, layout=..., backend=...)`.
Authoring workflows that only need to inspect or lower a model never pay the
cost of loading the C++ toolchain.

## Public API surface

The entries below are the stable public surface as of Spec 4. The top-level
`pops` namespace re-exports the most commonly used symbols.

| Symbol | Package |
|--------|---------|
| `pops.physics.Model` | `pops.physics` |
| `pops.time.Program` | `pops.time` |
| `pops.compile_problem(model=..., program=...)` | top-level (delegates to `pops.codegen`) |
| `pops.CompiledProblem` | `pops.codegen` |
| `pops.runtime.System` | `pops.runtime` (also `pops.System`) |
| `pops.runtime.AmrSystem` | `pops.runtime` (also `pops.AmrSystem`) |
| descriptor packages (`pops.numerics`, `pops.fields`, `pops.solvers`, `pops.mesh`, ...) | typed route selection |

## Public execution route

The public route is intentionally single:

```python
module = physics_model.to_module()
program = build_program(module)
compiled = pops.compile_problem(
    model=module,
    program=program,
    layout=layout,
    backend=backend,
)
sim = pops.System(layout=layout)
sim.install(compiled, instances=instances, params=params, solvers=solvers)
sim.step_cfl(cfl)
```

Strings may name user objects such as blocks or operators. They do not choose
algorithms, layouts, backends, solvers, limiters or output policies; those are
typed descriptors.

## Module file-size rule

Each module file must stay at or below 500 lines. Files that would exceed this
limit are split into mixins or sub-modules within the package.

## Related documentation

- Operator-first module reference: {doc}`../reference/operator-modules`
- Blackboard DSL (physics authoring): {doc}`../reference/board-like-dsl`
- Time program reference: {doc}`../reference/time-program`
- Symbolic DSL reference: {doc}`../reference/symbolic-dsl`
- Backend matrix: {doc}`../reference/backend-matrix`
