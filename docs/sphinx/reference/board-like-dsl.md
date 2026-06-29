# Physics authoring facade

`pops.physics` is the high-level way to write equations with physical names. It
is a facade over `pops.model`; it does not compile or run.

```{admonition} Boundary
:class: important
Python authors the model and program. C++/Kokkos/MPI executes the generated or
native route.
```

## Write physics

```python
from pops.physics import Model
from pops.math import sqrt, div, ddt

m = Model("isothermal_euler")
U = m.state("U", components=["rho", "mx", "my"], roles={"rho": "density"})
rho, mx, my = U

u = m.primitive("u", mx / rho)
v = m.primitive("v", my / rho)
cs2 = m.param("cs2", 1.0)
p = m.scalar("p", cs2 * rho)
c = m.scalar("c", sqrt(cs2))

F = m.flux(
    "F",
    on=U,
    x=[mx, mx * u + p, mx * v],
    y=[my, my * u, my * v + p],
    waves={"x": [u - c, u, u + c], "y": [v - c, v, v + c]},
)

explicit_rate = m.rate("explicit_rate", ddt(U) == -div(F))
module = m.lower()
```

## Use the model

```python
compiled = pops.compile_problem(model=module, time=program, backend=Production(), layout=layout)

sim = pops.System(n=mesh.n, L=mesh.L, periodic=mesh.periodic)
sim.install(
    compiled,
    instances={"plasma": {"model": module, "initial": U0, "spatial": spatial}},
)
sim.step_cfl(0.5)
```

## Time syntax

Use `pops.time.Program` for custom schedules:

```python
from pops.time import Program

T = Program("forward_euler").bind_operators(module)
U = T.state("U", block="plasma")
fields = T.call(fields_from_state, U.n)
rate = T.call(explicit_rate, U.n, fields)

T.define(U.next, U.n + T.dt * rate)
T.commit("plasma", U.next)
```

Use `pops.lib.time` for ready-made schemes.

## Operator-first layer

`pops.model.Module`, `OperatorHandle`, `Signature`, `Rate`, and
`pops.time.Program` remain first-class. They are the explicit layer for library
authors, tests, and inspection. The physics facade lowers to those objects.
