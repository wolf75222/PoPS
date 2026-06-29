# Public API contract

This page is the contract for the documented Python API. It is deliberately
smaller than the implementation surface: internal seams may exist because the
runtime needs them, but user documentation must not present them as alternate
front doors.

## Rule

Python describes a compiled problem. C++/Kokkos/MPI executes it.

The public flow is:

```python
case = pops.Case(layout=layout, name="run_name")
compiled = pops.compile(case, backend=Production())
sim = pops.bind(compiled, state=initial_state, params=params)
sim.run(t_end=1.0, cfl=0.4)
```

No Python callback is allowed in a cell loop, stage loop, Krylov loop, field
solve, halo exchange, reduction, AMR tag pass, regrid, reflux, output writer, or
checkpoint writer. If an operation is public, it must lower to a compiled C++
route.

## Public layers

| Package | Role |
| --- | --- |
| `pops.physics` | Physics authoring facade. It builds model/operator IR and lowers to `pops.model`. It does not compile or run. |
| `pops.model` | Operator-first model IR: states, fields, operators, handles, signatures, capabilities. |
| `pops.time` | Temporal language: `Program`, state-version handles, schedules, histories, and IR passes. |
| `pops.lib.time` | Ready-made time-program macros such as `ssprk2`, `ssprk3`, `rk4`, `strang`, and `bdf`. |
| `pops.mesh` | Mesh descriptors: `CartesianMesh`, `PolarMesh`, layouts, geometry, boundaries, AMR policies. |
| `pops.fields` | Elliptic field problems: `FieldProblem`, `PoissonProblem`, boundary conditions, RHS descriptors, coefficients. |
| `pops.numerics` | PDE discretisation descriptors: Riemann fluxes, reconstruction, limiters, RHS terms, projections. |
| `pops.linalg` | Algebraic problem descriptions: linear problems, matrix-free operators, norms, reductions. |
| `pops.solvers` | Compiled solver descriptors supplied by the core: elliptic, Krylov, Schur, preconditioners. |
| `pops.codegen` | C++ lowering, compile backends, optimization descriptors, inspection and generated-source access. |
| `pops.output` | Output/checkpoint policies. |
| `pops.external` | References to compiled external C++ bricks with manifests. |
| `pops.experimental` | Debug and experimental helpers. Not stable and not production documentation. |

`pops.lib` is for provided presets and ready-made assemblies. It is not a dump
for core primitives. In particular, solver descriptors live in `pops.solvers`,
not `pops.lib.solvers`.

## Typed choices

Strings name objects chosen by the user. Typed objects choose behavior.

Valid string uses:

```python
Model("euler")
case.block("ions", physics=ions)
RuntimeParam("nu", default=0.1)
PoissonProblem(name="phi", unknown="phi")
T.state("U", block="ions")
```

Invalid public style:

```python
# Do not document this style.
solver = "geometric_mg"
target = "amr_system"
method = "ssprk3"
kind = "runtime"
```

Documented style:

```python
from pops.codegen import Production
from pops.mesh.layouts import AMR, Uniform
from pops.solvers.elliptic import GeometricMG
from pops.lib.time import ssprk3
from pops.params import RuntimeParam, Positive

backend = Production()
solver = GeometricMG()
nu = RuntimeParam("nu", default=0.1, domain=Positive())
```

If a page needs to mention a lowered native token, it must describe it as an
internal lowering detail, not as user syntax.

## Handles, not string references

Strings may introduce names. Later references should use handles returned by
the authoring API.

```python
rate = model.rate("explicit_rate", equation=...)

T = Program("step")
U = T.state("U", block="plasma")
fields = T.call(fields_from_state, U.n)
R = T.call(rate, U.n, fields)
T.define(U.next, U.n + T.dt * R)
T.commit("plasma", U.next)
```

The current implementation still contains internal string tokens in lowering
helpers. They are not part of the documented user API.

## Time schemes

`pops.time` is the language. It contains `Program`, time-version handles,
schedules, histories, and analysis/optimization passes.

Ready schemes are in `pops.lib.time`:

```python
from pops.time import Program
from pops.lib.time import ssprk3

T = Program("advance")
ssprk3(T, "plasma")
```

For manual programs, use temporal handles:

```python
from pops.time import Program
from pops.model import OperatorHandle

T = Program("ssprk3_manual")
U = T.state("U", block="plasma")
T.bind_operators(model)

fields_from_state = OperatorHandle("fields_from_state", kind="field_operator")
rate = model.rate_operator("explicit_rate", flux=True, sources=["default"])

f0 = T.call(fields_from_state, U.n)
k0 = T.call(rate, U.n, f0)
T.define(U.stage(1), U.n + T.dt * k0)

f1 = T.call(fields_from_state, U.stage(1))
k1 = T.call(rate, U.stage(1), f1)
T.define(U.stage(2), 0.75 * U.n + 0.25 * (U.stage(1) + T.dt * k1))

f2 = T.call(fields_from_state, U.stage(2))
k2 = T.call(rate, U.stage(2), f2)
T.define(U.next, (1.0 / 3.0) * U.n + (2.0 / 3.0) * (U.stage(2) + T.dt * k2))

T.commit("plasma", U.next)
```

`U.n`, `U.stage(k)`, `U.next`, and `U.prev` are IR handles. They do not hold
runtime arrays.

## Mesh layouts

The mesh layout is not a backend and not a target string.

```python
from pops.mesh.cartesian import CartesianMesh
from pops.mesh.layouts import Uniform, AMR
from pops.mesh.amr import Refine, RegridEvery, PatchLayout

mesh = CartesianMesh(n=128, L=1.0, periodic=True)
uniform = Uniform(mesh)

amr = AMR(
    mesh,
    max_levels=2,
    ratio=2,
    regrid=RegridEvery(8),
    patches=PatchLayout(coarse_max_grid=32),
    refine=Refine.on("density").above(0.05),
)
```

`pops.compile` derives the runtime route from the layout. The user does not pass
a target.

## AMR compatibility

AMR is part of the public assembly model, not a separate scripting path. A
feature documented for `Case` must either work on both `Uniform` and `AMR`, or
declare a precise mathematical incompatibility in its descriptor. A missing
Python binding, missing codegen branch, or missing runtime plumbing is not a
valid public limitation.

The AMR route uses:

- typed `AMR` layout descriptors;
- typed refinement and patch policies;
- per-block compiled model loaders;
- runtime C++/Kokkos/MPI data structures;
- C++ tag, regrid, halo, reflux, field, output, and diagnostic paths.

No AMR page should instruct users to instantiate `AmrSystem` directly as the
front door.

## Inspection

Inspectable objects should print concise summaries and expose structured
reports:

```python
print(case)
print(compiled)
print(sim)

compiled.inspect()
compiled.arguments()
compiled.estimate_memory(grid=mesh, platform=...)
compiled.dump_ir()
compiled.dump_cpp()
```

Profiling is off by default:

```python
with sim.profile(pops.Profile.Advanced()) as prof:
    sim.run(t_end=1.0, cfl=0.4)

prof.summary().print()
```

A report may say a counter is unavailable for the current build. It must not
fake a zero.
