# First run

This page shows the public Python flow:

1. describe physics with `pops.physics.Model`;
2. describe the mesh layout and field problem with typed descriptors;
3. build a time `Program`;
4. compile the `Case`;
5. bind runtime data;
6. run in C++.

Python builds descriptors and IR. It does not run cell loops.

```python
import numpy as np
import pops
from pops.codegen import Production
from pops.fields import PoissonProblem
from pops.fields.bcs import Periodic
from pops.fields.rhs import ChargeDensity
from pops.lib.time import ssprk3
from pops.math import ddt, div, grad, laplacian
from pops.mesh.cartesian import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.numerics.riemann import Rusanov
from pops.numerics.reconstruction.limiters import Minmod
from pops.physics import Model
from pops.solvers.elliptic import GeometricMG
from pops.time import Program

n = 96
xs = (np.arange(n) + 0.5) / n
X, Y = np.meshgrid(xs, xs)
y0 = 0.5 + 0.02 * np.cos(2.0 * np.pi * 2.0 * X)
ne0 = np.ascontiguousarray(1.0 + np.exp(-((Y - y0) ** 2) / 0.05 ** 2))

m = Model("diocotron")
U = m.state("U", components=["ne"], roles={"ne": "density"})
(ne,) = U

phi = m.field("phi")
m.solve_field(
    "fields_from_state",
    equation=(-laplacian(phi) == ne),
    outputs={"phi": phi, "grad_x": grad(phi).x, "grad_y": grad(phi).y},
    solver=GeometricMG(),
)

E = m.vector_field("E", x=-grad(phi).x, y=-grad(phi).y)
flux = m.flux(
    "F",
    on=U,
    x=[ne * E.y],
    y=[ne * (-E.x)],
    waves={"x": [E.y], "y": [-E.x]},
)
m.rate("explicit_rate", ddt(U) == -div(flux))
m.check()

poisson = PoissonProblem(
    name="phi",
    unknown="phi",
    equation=(-laplacian("phi") == ChargeDensity.from_blocks("ne")),
    bcs=(Periodic(),),
    solver=GeometricMG(),
)

time = Program("advance")
ssprk3(time, "ne")

mesh = CartesianMesh(n=n, L=1.0, periodic=True)
case = (
    pops.Case(layout=Uniform(mesh), name="diocotron")
    .block("ne", physics=m, spatial=pops.FiniteVolume(limiter=Minmod(), riemann=Rusanov()))
    .field(poisson)
    .time(time)
)

compiled = pops.compile(case, backend=Production())
sim = pops.bind(compiled, state={"ne": ne0})
sim.run(t_end=0.1, cfl=0.4)

print("t       =", sim.time())
print("mass    =", sim.mass("ne"))
print("density =", sim.density("ne").shape)
```

## What is public

- `pops.Case` is the top-level assembly.
- `pops.compile` lowers the assembly and chooses the runtime route from the layout.
- `pops.bind` attaches arrays, parameters, aux fields, field solvers, outputs, and checkpoint policies.
- `sim.run` advances the compiled C++ runtime.

Runtime classes such as `System` and `AmrSystem` may still exist internally and in tests. They are
not the documented authoring path.

## AMR

To run the same case on AMR, keep the model and time program and change the layout:

```python
from pops.mesh.amr import PatchLayout, Refine, RegridEvery
from pops.mesh.layouts import AMR

layout = AMR(
    mesh,
    max_levels=2,
    ratio=2,
    regrid=RegridEvery(8),
    patches=PatchLayout(coarse_max_grid=32),
    refine=Refine.on("density").above(0.05),
)
```

Then build the case with `layout=layout`. The public flow remains `Case -> compile -> bind -> run`.
