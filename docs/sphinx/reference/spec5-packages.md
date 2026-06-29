# Package map

This page maps the public Python packages after the Spec 5/6 cleanup. The
source of truth for the rules is [public API contract](public-api-contract.md).

## Authoring packages

| Package | Purpose |
| --- | --- |
| `pops.physics` | User-facing physics authoring facade. Produces model/operator IR. |
| `pops.model` | Typed operator-first core: states, fields, signatures, operator handles. |
| `pops.time` | Time-program language: `Program`, version handles, schedules, histories, passes. |
| `pops.fields` | Field problems and elliptic authoring descriptors. |
| `pops.mesh` | Meshes, layouts, geometry, boundaries, AMR policies. |
| `pops.numerics` | Discretisation descriptors: Riemann, reconstruction, RHS terms, projections. |
| `pops.linalg` | Algebraic problem descriptions. |

## Runtime and compile packages

| Package | Purpose |
| --- | --- |
| `pops.codegen` | C++ lowering, build/cache, optimization descriptors, inspection. |
| `pops.solvers` | Provided compiled solver descriptors. |
| `pops.output` | Output and checkpoint policies. |
| `pops.external` | Compiled external C++ brick manifests and references. |
| `pops.experimental` | Debug and experimental tools, not production docs. |

## Presets

`pops.lib` contains ready-made presets. Today that includes:

- `pops.lib.time`: ready time-program macros;
- `pops.lib.models`: provided model assemblies, including moment-model presets.

Do not add core primitives to `pops.lib`. A primitive that users compose with
other pieces belongs in the matching top-level package.

## Important ownership rules

- `pops.time` is not the home of ready schemes. Use `pops.lib.time`.
- `pops.solvers` is the public home of solver descriptors. Do not document
  `pops.lib.solvers`.
- `pops.numerics` owns Riemann, reconstruction, limiters, projections, and RHS
  terms.
- `pops.mesh.layouts` owns `Uniform` and `AMR`; these are layouts, not targets.
- `pops.physics` lowers to model IR. It does not own compilation or runtime
  execution.

## Example

```python
from pops.codegen import Production
from pops.lib.time import ssprk3
from pops.mesh.cartesian import CartesianMesh
from pops.mesh.layouts import AMR
from pops.mesh.amr import Refine, RegridEvery
from pops.solvers.elliptic import GeometricMG
from pops.time import Program

mesh = CartesianMesh(n=128, L=1.0, periodic=True)
layout = AMR(mesh, max_levels=2, ratio=2, regrid=RegridEvery(8),
             refine=Refine.on("density").above(0.05))

T = Program("advance")
ssprk3(T, "plasma")

compiled = pops.compile_problem(model=module, program=T, backend=Production(), layout=layout)
```

The only strings above are names chosen by the user: block names, field names,
operator names, and parameter names.
