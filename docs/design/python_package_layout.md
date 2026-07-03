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
| `pops.solvers` | Linear / nonlinear / Schur / elliptic solver and preconditioner descriptors. |
| `pops.fields` | Elliptic field-problem authoring (`FieldProblem` / `PoissonProblem`) and its catalog. |
| `pops.diagnostics` | Diagnostic descriptors (norms, reductions). |
| `pops.params` | Typed runtime-parameter descriptors. |
| `pops.output` | Output / checkpoint policy descriptors. |
| `pops.external` | Inert descriptors for external integrations. |
| `pops.lib` | Ready-to-use presets only: `lib.time` schemes, `lib.models` models, `lib.presets` bundles. |
| `pops.codegen` | Lowering / build toolchain and the internal `compile_problem` driver + `@solver` DSL. |
| `pops.runtime` | The runtime: `System` / `AmrSystem`, bind adapters, doctor, bricks. The ONLY layer that imports `_pops`. |
| `pops.experimental` | Tests-only, non-stable host backends (`PythonFlux`); never on the public surface. |

Nested sub-packages keep the same rules: `pops.mesh.{layouts,amr,boundaries,geometry,masks}`,
`pops.numerics.{riemann,reconstruction,terms,variables,projections}`,
`pops.solvers.{elliptic,krylov,nonlinear,schur}`, `pops.moments.closures`,
`pops.codegen.solvers`, `pops.runtime.amr`, `pops.lib.{time,models,models.moments,presets}`.

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
is a leaf that composes descriptors: a preset in `lib.presets` pairs a `lib.models` model with a
`lib.time` scheme and never reaches up into `codegen` or `runtime`.

## Public front door

`pops.compile` / `pops.bind` are the only public compile/bind entry points (ADC-523). The low-level
`compile_problem` driver and the concrete `CompiledProblem` loader class are advanced/internal,
reachable as `pops.codegen.compile_problem` / `pops.codegen.CompiledProblem`; annotate the compiled
handle against the `pops.CompiledArtifact` protocol.

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

## Root kernel modules (kept flat on purpose)

A few modules stay at the package root and are deliberately NOT buried in a package. They are the
shared descriptor kernel, the bootstrap, and the top-level assembly that the whole tree depends on;
the import-graph test treats them as untracked (not a layer), so moving them into a package would
create edges and risk cycles across roughly forty importers. Do not relocate them:

- `pops/__init__.py` -- the runtime facade (re-exports the public surface).
- `descriptors.py` -- the base `Descriptor`, imported by mesh / numerics / fields / moments / ...
- `math.py` -- small math helpers imported by time / params / fields.
- `case.py` -- the top-level `Case` assembly (a root module, not a package).
- `runtime_environment.py` -- the runtime-environment report, imported by mesh and runtime.
- `_bootstrap.py`, `_version.py`, `_descriptor_protocol.py`, `_capabilities*.py` -- bootstrap,
  version, the descriptor protocol, and the capability inspection kernel.

A future change that wants to package one of these must first redesign `test_import_graph.py` to
track it as a layer, so the new edges are checked rather than hidden.
