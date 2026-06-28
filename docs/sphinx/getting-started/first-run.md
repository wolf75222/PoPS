# First run

The smallest `pops` program through the public API: we author a diocotron model as formulas, we
declare the typed Poisson field, we assemble a `pops.Case`, we compile it and bind a runnable
simulation, then we advance and read the density back. Copyable as is (it assumes only that the
module is [installed](installation.md) and importable).

```python
import numpy as np
import pops
import pops.time as T
from pops.physics import Model
from pops.math import laplacian, grad, div, ddt
from pops.mesh.cartesian import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.fields import PoissonProblem
from pops.fields.bcs import Periodic
from pops.fields.rhs import ChargeDensity
from pops.solvers.elliptic import GeometricMG
from pops.codegen import Production

# 1. The physics: a scalar density advected by the E x B drift, coupled to Poisson.
m = Model("diocotron")
U = m.state("U", components=["ne"], roles={"ne": "density"})
(ne,) = U
phi = m.field("phi")
m.solve_field("fields_from_state",
              equation=(-laplacian(phi) == ne),          # rhs = charge density
              outputs={"phi": phi, "grad_x": grad(phi).x, "grad_y": grad(phi).y},
              solver="geometric_mg")
E = m.vector_field("E", x=-grad(phi).x, y=-grad(phi).y)
flux = m.flux("F", on=U, x=[ne * E.y], y=[ne * (-E.x)],   # v = (E_y, -E_x)
              waves={"x": [E.y], "y": [-E.x]})
m.rate("explicit_rate", ddt(U) == -div(flux))
m.check()

# 2. The typed elliptic field: -laplacian(phi) == charge density, multigrid solver.
poisson = PoissonProblem(
    name="phi", unknown="phi",
    equation=(-laplacian("phi") == ChargeDensity.from_blocks("ne")),
    bcs=(Periodic(),),
    solver=GeometricMG(),
)

# 3. The assembly: a periodic 96 x 96 square, one block, the field, the time scheme.
case = (pops.Case(layout=Uniform(CartesianMesh(n=96, L=1.0, periodic=True)), name="diocotron")
        .block("ne", physics=m)
        .field(poisson)
        .time(T.Program("euler")))

# 4. Initial condition: a perturbed charge band along x.
n = 96
xs = (np.arange(n) + 0.5) / n
X, Y = np.meshgrid(xs, xs)                        # indexing 'xy': ne[j, i]
y0 = 0.5 + 0.02 * np.cos(2.0 * np.pi * 2.0 * X)   # azimuthal mode 2
ne0 = 1.0 + np.exp(-((Y - y0) ** 2) / 0.05 ** 2)

# 5. Compile the assembly and bind a runnable simulation, then advance a few CFL steps.
compiled = pops.compile(case, backend=Production())
sim = pops.bind(compiled, state={"ne": np.ascontiguousarray(ne0)})
sim.run(0.1, cfl=0.4)

print("t        =", sim.time())
print("mass     =", sim.mass("ne"))             # conserved by the periodic advective transport
print("density  =", sim.density("ne").shape)    # (96, 96)
```

What the key calls do:

- `pops.physics.Model(...)` authors the physics as formulas (state, flux, the elliptic
  `solve_field`); it lowers to the typed operator IR `pops.compile` consumes.
- `PoissonProblem(unknown=, equation=, solver=, bcs=)` is the typed elliptic surface: the unknown
  field, its governing equation `-laplacian(phi) == rhs`, the solver
  (`pops.solvers.elliptic.GeometricMG()` or `FFT()`), and the boundary conditions. Attach it to the
  case with `case.field(poisson)`.
- `pops.Case(layout=Uniform(mesh)).block(...).field(...).time(...)` assembles the inert,
  typed description. `pops.compile(case, backend=Production())` lowers it to a compiled handle;
  `pops.bind(compiled, state=, params=, aux=, solvers=)` wires a runnable simulation.
- `sim.run(t_end, cfl=)` advances; `density(name)` / `mass(name)` / `time()` read the state.

```{note}
The native `System` runtime methods (`add_block`, `add_equation`, `set_poisson`, `step_cfl`) are
the low-level seam `pops.bind` builds on; they remain available for the native/AMR runtime and the
tests, but the documented public front door is the `Case -> compile -> bind -> run` flow above.
```

To author a richer model, refine on an adaptive hierarchy (`pops.mesh.layouts.AMR`), and produce
the figures and the GIF, follow the [A->Z tutorial](tutorial.md).
