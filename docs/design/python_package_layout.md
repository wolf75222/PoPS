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
| `pops.math` | Public symbolic expressions and operators used by scientific authoring. |
| `pops._ir` | Internal symbolic IR consumed by `pops.math`, validation and lowering; never a user import surface. |
| `pops.identity` | Canonical semantic and artifact identities. |
| `pops.frames` | Coordinate frames and typed axes. |
| `pops.domain` | Physical domains, extents and topological boundaries. |
| `pops.representations` | State/field representation descriptors. |
| `pops.spaces` | Generic authoring-space descriptors. |
| `pops.model` | The operator-first `Module` and its typed operator registry. |
| `pops.problem` | `Case` assembly, qualified instance handles and immutable snapshots. |
| `pops.physics` | The physics writing facade (`pops.physics.Model`) that lowers to a `Module`. |
| `pops.time` | The time-program language (`Program`, solve/commit/control-flow IR). |
| `pops.initial` | Initial-condition authoring protocols. |
| `pops.moments` | The moment-model construction kit and closures. |
| `pops.mesh` | Grid topology, boundary, geometry, mask and layout-plan descriptors. |
| `pops.amr` | Public adaptive-layout authoring and materialization protocols. |
| `pops.layouts` | Public `Uniform` / `AMR` layout providers. |
| `pops.boundary` | Physical transport-boundary authoring. |
| `pops.numerics` | Discretization descriptors: Riemann fluxes, reconstruction, terms, variables, projections. |
| `pops.projection` | Generic projection descriptors. |
| `pops.linalg` | Abstract algebra descriptors (`A x = b`, operators, norms, reductions). |
| `pops.solvers` | Executable linear / nonlinear / elliptic solver and preconditioner descriptors, plus hierarchy providers. |
| `pops.fields` | Physical `FieldOperator` declarations, numerical `FieldDiscretization` plans, outputs and field-read policies. |
| `pops.diagnostics` | Diagnostic descriptors (norms, reductions). |
| `pops.params` | Typed runtime-parameter descriptors. |
| `pops.output` | `ConsumerGraph`, direct scientific-output, checkpoint, format and writer descriptors. |
| `pops.external` | Authenticated source and fixed-binary component package contracts. |
| `pops.lib` | Ready-to-use implementations only: `lib.time`, `lib.models`, `lib.initial`, `lib.amr`. |
| `pops.codegen` | Public compiler-provider protocols plus internal lowering/build providers consumed by `pops.compile`. |
| `pops.runtime` | Internal `RuntimeInstance`, execution, restart publication and diagnostics; the package exports no user symbols. |
| `pops.experimental` | Tests-only, non-stable host backends (`PythonFlux`); never on the public surface. |

`pops.runtime` is an opaque implementation package, not an author or provider namespace. Its
execution leaf modules use private names such as `_runtime_instance`, `_consumer` and
`_output_publisher`; the former paths `pops.runtime.runtime_instance`, `pops.runtime.consumer` and
`pops.runtime.output_publisher` do not exist and have no compatibility shims. Scientific-output
extensions instead implement the contracts exported by `pops.output`.

Nested sub-packages keep the same rules: `pops.mesh.{_amr,boundaries,geometry,masks}`,
`pops.numerics.{riemann,reconstruction,terms,variables,projections}`,
`pops.solvers.{elliptic,krylov,nonlinear}`, `pops.moments.closures`,
`pops.codegen.solvers`, `pops.runtime.amr`,
`pops.lib.{time,models,models.moments,initial,amr}`.

## Layering DAG

The sub-packages form a directed acyclic dependency stack. A package may import only the layers
listed as ALLOWED for it (this is the single source in
`tests/python/architecture/test_import_graph.py`):

```
_ir, identity, representations, spaces, projection, params, linalg, experimental -> (nothing)
frames      -> identity
domain      -> frames, identity
model       -> _ir, identity, params
problem     -> _ir, identity, model
physics     -> _ir, model, problem
time        -> _ir, model, params
initial     -> model
mesh        -> domain, frames, identity, model, params
amr         -> _ir, identity, mesh, model, time
layouts     -> amr, mesh
boundary    -> _ir, domain, identity, model, representations
numerics    -> identity, model, params
solvers     -> identity
fields      -> _ir, identity, model, time
moments     -> _ir
diagnostics -> linalg
output      -> identity, model, time
external    -> identity, model
lib         -> fields, frames, moments, params, physics, solvers, time
codegen     -> _ir, fields, identity, model, params, solvers, time
runtime     -> _ir, codegen, fields, identity, mesh, model, output, time
```

The import-time DAG above is exhaustive. Native-extension loads are checked at every lexical scope
and belong only to explicit phase cuts: `_bootstrap`, runtime providers, the production toolchain,
external-component loading and the runtime-environment report. Importing authoring/codegen modules
remains `_pops`-free; invoking `compile`, `bind`, component load or a runtime report may cross that
named boundary. `pops.lib`
is a leaf that composes authenticated handles into ordinary Programs: every `lib.time` factory
returns the same `pops.Program` type as explicit authoring and never reaches into `codegen` or
`runtime`.

## Public front door

The root lifecycle is exactly `validate`, `resolve`, `compile`, `bind`, and `run`; `inspect` and
`explain` are its pure reporting operations. Resolved plans, compiled
artifact records, bind-input evidence and install plans are phase-internal values: users receive
them from the lifecycle but never import or construct their concrete codegen classes. External AOT
components use the single `pops.external.load(...).require(...)` then
`pops.external.compile_component(...)` package contract.

## Rules

- **No flat monolith.** A responsibility large enough to be a package is a package, never a single
  root `.py` file. `dsl`, `model`, `time`, `physics`, `lib`, `library`, `moments` and `integrate`
  must not exist as root modules (`test_no_flat_modules.py`); `std` / `custom` escape hatches and a
  flat `models` package are banned outright (`test_no_forbidden_paths.py`).
- **Cohesive module ownership.** `test_file_sizes.py` enforces canonical importable module paths and
  a declarative root facade. Physical line count is not used as a proxy for design quality; split a
  module when responsibilities or ownership diverge, not to satisfy an arbitrary threshold.
- **Acyclic + layered at import time.** Module-scope imports must respect the exhaustive DAG above;
  a cross-layer edge to a non-allowed layer, a missing package, or any cycle fails
  `test_import_graph.py`. A lazy import is permitted only at a real phase boundary; it must not be a
  compatibility alias or a way to disguise ownership. Native-extension loads have a separate
  all-scopes allowlist.

## Root kernel modules and assembly package

A few modules stay at the package root, while the larger assembly lives in `pops.problem`. They are
the shared descriptor kernel, bootstrap and top-level ownership types that the whole tree depends on;
the import-graph test treats them as untracked (not a layer). Do not relocate them without first
updating the dependency model:

- `pops/__init__.py` -- the lazy public lifecycle facade.
- `descriptors.py` -- the base `Descriptor`, imported by mesh / numerics / fields / moments / ...
- `math.py` -- the public symbolic facade over the private `_ir` package.
- `problem/` -- the top-level `Case` assembly, qualified handles and immutable snapshots.
- `runtime_environment.py` -- the runtime-environment report, imported by mesh and runtime.
- `_bootstrap.py`, `_version.py`, `_descriptor_protocol.py`, `_capabilities*.py` -- bootstrap,
  version, the descriptor protocol, and internal capability/report support.

A future change that wants to package one of these must first redesign `test_import_graph.py` to
track it as a layer, so the new edges are checked rather than hidden.
