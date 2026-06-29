# Symbolic physics DSL

`pops.physics.Model` is the physics authoring facade. It lets users write
states, fields, fluxes, sources, rates, projections, and capabilities with
Python objects. The facade builds model/operator IR. It does not run kernels and
is not the public compile front door.

The public route is:

```python
model = Model("name")
module = model.to_module()
compiled = pops.compile_problem(model=module, program=program, backend=Production(), layout=layout)
sim = pops.System(n=mesh.n, L=mesh.L, periodic=mesh.periodic)
sim.install(compiled, instances={"block": {"model": module, "initial": state}})
```

## State

```python
from pops.physics import Model

m = Model("euler")
U = m.state("U", components=["rho", "mx", "my", "E"],
            roles={"rho": "density", "mx": "momentum_x", "my": "momentum_y", "E": "energy"})
rho, mx, my, E = U
```

Strings in this block are user names and roles. They are not algorithm
selectors.

## Fields

```python
from pops.math import grad, laplacian
from pops.solvers.elliptic import GeometricMG

phi = m.field("phi")
m.solve_field(
    "fields_from_state",
    equation=(-laplacian(phi) == rho),
    outputs={"phi": phi, "grad_x": grad(phi).x, "grad_y": grad(phi).y},
    solver=GeometricMG(),
)
```

Field problems attached to the case use `pops.fields` descriptors. The model
field declaration and the case-level `PoissonProblem` must agree on names and
outputs.

## Flux and rates

```python
from pops.math import ddt, div

flux = m.flux(
    "euler_flux",
    on=U,
    x=[mx, ...],
    y=[my, ...],
    waves={"x": [...], "y": [...]},
)

rate = m.rate("explicit_rate", ddt(U) == -div(flux))
```

The object returned by `m.rate` can be used as an operator handle by a time
program. Programs should reference handles, not free string operator names.

## Sources and local operators

```python
source = m.source("electric", on=U, value=[...])
L = m.local_linear_operator("lorentz", on=U, matrix=[[...], [...], [...]])
```

Use typed RHS terms or operator handles in `pops.time.Program`.

## Parameters

Use typed parameter descriptors when the parameter is part of public assembly:

```python
from pops.params import RuntimeParam, Positive

nu = RuntimeParam("nu", default=0.1, domain=Positive())
compiled = pops.compile_problem(model=module, program=program, backend=Production(), layout=layout)
sim.install(compiled, params={"nu": 0.2}, instances={"block": {"model": module, "initial": state}})
```

The parameter name is a string because the user names it. The runtime/const
choice is a typed object.

## Capabilities

The model must declare what numerical routes need:

- physical flux;
- wave speeds;
- pressure/contact structure for HLLC;
- Roe data when Roe is requested;
- positivity/realizability projection when a scheme needs it;
- named fields and aux channels.

Descriptor validation should reject an incompatible route before runtime.

## Lowering boundary

`pops.physics` writes model/operator IR. It does not own code generation,
runtime allocation, MPI, Kokkos execution, AMR, profiling, or output.

Compilation is owned by `pops.compile_problem` / `pops.codegen`. Execution is
owned by the explicit runtime facade after `sim.install(compiled, ...)`.
