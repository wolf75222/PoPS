# Python package layout

Canonical map of the `python/pops` tree: what each package owns, how the packages may depend on
one another, and the rules that keep the tree flat and acyclic. This is the durable reference for
anyone adding or moving a Python module. It is enforced by source-only tests under
`tests/python/architecture/` (no compiled `_pops` needed), so a violation fails CI in the bare lane.

## Package map

Each top-level package is a single responsibility. Concrete models and ready-made schemes live under
`pops.lib`; the generic building blocks live in the central packages.

| Package | Responsibility |
|---|---|
| `pops.ir` | Symbolic IR (expressions, values, nodes). Imports nothing else in `pops`. |
| `pops.model` | The operator-first `Module` and its typed operator registry. |
| `pops.physics` | The physics writing facade (`pops.physics.Model`) that lowers to a `Module`. |
| `pops.time` | The time-program language (`Program`, solve/commit/control-flow IR). |
| `pops.moments` | The moment-model construction kit and closures. |
| `pops.mesh` | Mesh, layout (`Uniform` / `AMR`), boundary, geometry and mask descriptors. |
| `pops.numerics` | Discretization descriptors: Riemann fluxes, reconstruction, terms, variables, projections. |
| `pops.linalg` | Abstract algebra descriptors (`A x = b`, operators, norms, reductions). |
| `pops.solvers` | Executable linear / nonlinear / elliptic solver and preconditioner descriptors, plus hierarchy providers. |
| `pops.fields` | Physical `FieldOperator` declarations, numerical `FieldDiscretization` plans, outputs and field-read policies. |
| `pops.diagnostics` | Diagnostic descriptors (norms, reductions). |
| `pops.params` | Typed runtime-parameter descriptors. |
| `pops.output` | Direct scientific-output, checkpoint, format and writer descriptors. |
| `pops.external` | Authenticated source and fixed-binary component package contracts. |
| `pops.lib` | Ready-to-use implementations only: `lib.time`, `lib.models`, `lib.initial`, `lib.amr`. |
| `pops.codegen` | Internal lowering/build providers consumed by the public `pops.compile` phase. |
| `pops.runtime` | Unified `RuntimeInstance`, execution, restart, consumers and diagnostics. The ONLY layer that imports `_pops`. |
| `pops.experimental` | Tests-only, non-stable host backends (`PythonFlux`); never on the public surface. |

Nested sub-packages keep the same rules: `pops.mesh.{layouts,amr,boundaries,geometry,masks}`,
`pops.numerics.{riemann,reconstruction,terms,variables,projections}`,
`pops.solvers.{elliptic,krylov,nonlinear}`, `pops.moments.closures`,
`pops.codegen.solvers`, `pops.runtime.amr`,
`pops.lib.{time,models,models.moments,initial,amr}`.

## Layering DAG

The sub-packages form a directed acyclic dependency stack. A package may import only the layers
listed as ALLOWED for it (this is the single source in
`tests/python/architecture/test_import_graph.py`):

```
ir          -> (nothing)
model       -> ir
physics     -> ir, model
time        -> ir, model
mesh        -> (nothing)
numerics    -> (nothing)
linalg      -> (nothing)
solvers     -> (nothing)
fields      -> (nothing)
moments     -> ir
diagnostics -> linalg
params      -> (nothing)
output      -> (nothing)
external    -> (nothing)
lib         -> ir, model, time, physics, moments
codegen     -> ir, model, physics, time, lib, solvers
runtime     -> ir, model, physics, time, lib, mesh, codegen   (and _pops)
```

Only `pops.runtime` imports the compiled `_pops` extension. `codegen` and everything below stay
`_pops`-free (so the codegen and authoring surface imports without the native build). `pops.lib`
is a leaf that composes authenticated handles into ordinary Programs: every `lib.time` factory
returns the same `pops.Program` type as explicit authoring and never reaches into `codegen` or
`runtime`.

## Public front door

The root lifecycle is exactly `validate`, `resolve`, `compile`, `bind`, and `run`; `inspect` and
`explain` are its pure reporting operations. Resolved plans, compiled
artifact records, bind-input evidence and install plans are phase-internal values: users receive
them from the lifecycle but never import or construct their concrete codegen classes. External AOT
components use the independent `pops.external.load(...).require(...)` and
`pops.codegen.compile_component(...)` package contract.

## Rules

- **No flat monolith.** A responsibility large enough to be a package is a package, never a single
  root `.py` file. `dsl`, `model`, `time`, `physics`, `lib`, `library`, `moments` and `integrate`
  must not exist as root modules (`test_no_flat_modules.py`); `std` / `custom` escape hatches and a
  flat `models` package are banned outright (`test_no_forbidden_paths.py`).
- **500-line budget.** Every `python/pops/**/*.py` stays at or under 500 lines
  (`test_file_sizes.py`); the sole exception is the facade `pops/__init__.py`, capped at 120 lines
  (it re-exports, it does not implement).
- **Acyclic + layered.** Module-scope imports must respect the DAG above; a cross-layer edge to a
  non-allowed layer, or any cycle, fails `test_import_graph.py`. Push a cross-layer dependency into
  a function-scope (lazy) import when the module-scope edge would violate the layering.

## Root kernel modules and assembly package

A few modules stay at the package root, while the larger assembly lives in `pops.problem`. They are
the shared descriptor kernel, bootstrap and top-level ownership types that the whole tree depends on;
the import-graph test treats them as untracked (not a layer). Do not relocate them without first
updating the dependency model:

- `pops/__init__.py` -- the lazy public lifecycle facade.
- `descriptors.py` -- the base `Descriptor`, imported by mesh / numerics / fields / moments / ...
- `math.py` -- small math helpers imported by time / params / fields.
- `problem/` -- the top-level `Case` assembly, qualified handles and immutable snapshots.
- `runtime_environment.py` -- the runtime-environment report, imported by mesh and runtime.
- `_bootstrap.py`, `_version.py`, `_descriptor_protocol.py`, `_capabilities*.py` -- bootstrap,
  version, the descriptor protocol, and internal capability/report support.

A future change that wants to package one of these must first redesign `test_import_graph.py` to
track it as a layer, so the new edges are checked rather than hidden.
