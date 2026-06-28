# Write a model with the physics DSL

Use `pops.physics.Model` when you want to author equations directly instead of
choosing a ready-made model from `pops.lib.models`.

`pops.physics` lowers to `pops.model`. It does not compile by itself and it does
not run numerical loops.

## Declare fields and rates

```python
from pops.physics import Model
from pops.math import div, ddt, grad, laplacian
from pops.solvers.elliptic import GeometricMG

m = Model("diocotron")
U = m.state("U", components=["ne"], roles={"ne": "density"})
ne = U[0]

phi = m.field("phi")
m.solve_field(
    "fields_from_state",
    equation=(-laplacian(phi) == ne),
    outputs={"phi": phi, "grad_x": grad(phi).x, "grad_y": grad(phi).y},
    solver=GeometricMG(),
)

E = m.vector_field("E", x=-grad(phi).x, y=-grad(phi).y)
F = m.flux("F", on=U, x=[ne * E.y], y=[ne * (-E.x)], waves={"x": [E.y], "y": [-E.x]})
explicit_rate = m.rate("explicit_rate", ddt(U) == -div(F))

model = m.lower()
```

The names `U`, `ne`, `phi`, and `explicit_rate` are user-facing identifiers. The
solver choice is the typed `GeometricMG()` descriptor.

## Compose with numerical descriptors

```python
from pops.numerics.riemann import Rusanov
from pops.numerics.reconstruction import MUSCL
from pops.numerics.reconstruction.limiters import Minmod

spatial = pops.FiniteVolume(
    riemann=Rusanov(),
    reconstruction=MUSCL(limiter=Minmod()),
)
```

## Compile through a case

```python
from pops.mesh import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.time import Program
from pops.lib.time import ssprk2
from pops.codegen import Production

program = Program("ssprk2")
ssprk2(program, "ne")

case = (
    pops.Case(layout=Uniform(CartesianMesh(n=96, L=1.0, periodic=True)))
    .block("ne", physics=model, spatial=spatial)
    .time(program)
)

compiled = pops.compile(case, backend=Production())
sim = pops.bind(compiled, state={"ne": ne0})
sim.run(t_end=0.1, cfl=0.4)
```

Use `layout=AMR(...)` for adaptive runs. The model and time program stay the
same when their descriptors declare AMR support.
