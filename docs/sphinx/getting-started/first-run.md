# First run

This page shows the public Python flow:

1. describe physics with `pops.physics.Model`;
2. describe the mesh layout and field problem with typed descriptors;
3. build a time `Program`;
4. compile the model + program into a compiled problem artifact;
5. install that artifact on a `System` with runtime data;
6. run in C++.

Python builds descriptors and IR. It does not run cell loops.

```python
import numpy as np
import pops
from pops.codegen import Production
from pops.lib.time import ssprk3
from pops.math import ddt, div, grad, laplacian
from pops.mesh.cartesian import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.numerics.riemann import Rusanov
from pops.numerics.spatial import spatial
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

time = Program("advance")
ssprk3(time, "ne")

mesh = CartesianMesh(n=n, L=1.0, periodic=True)
layout = Uniform(mesh)
module = m.to_module()

compiled = pops.compile_problem(model=module, program=time, backend=Production(), layout=layout)
sim = pops.System(n=n, L=1.0, periodic=True)
sim.install(
    compiled,
    instances={
        "ne": {
            "model": module,
            "initial": ne0,
            "spatial": spatial.FiniteVolume(reconstruction=Minmod(), riemann=Rusanov()),
        },
    },
    solvers={"phi": GeometricMG()},
)
while sim.time() < 0.1:
    sim.step_cfl(0.4)

print("t       =", sim.time())
print("mass    =", sim.mass("ne"))
print("density =", sim.density("ne").shape)
```

## What is public

- `pops.compile_problem` lowers the typed model and time program into one compiled problem artifact.
- `pops.System(layout=...)` is the explicit runtime facade for uniform and AMR layouts.
- `sim.install(compiled, ...)` attaches arrays, parameters, aux fields, field solvers, outputs, and checkpoint policies.
- `sim.step_cfl` advances the compiled C++ runtime.

Low-level runtime setters remain private implementation details. Public user code enters through
`sim.install(compiled, ...)`.

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

Then compile with `layout=layout` and install the artifact on the matching runtime. The public flow
remains `compile_problem -> System(layout=...) -> install -> step_cfl`.
