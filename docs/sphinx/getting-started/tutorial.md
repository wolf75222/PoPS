# A-to-Z tutorial

This tutorial is the longer version of the first run. It keeps one rule: every
user-visible operation goes through the typed assembly path.

```text
pops.physics.Model
    -> pops.Case
    -> pops.compile
    -> pops.bind
    -> sim.run
```

The older runtime methods are implementation seams and tests. They are not the
documented tutorial path.

## Build and import

Use the repository scripts for the Python module:

```bash
bash scripts/setup_env.sh
bash scripts/build_python.sh
python -c "import pops; pops.doctor()"
```

For backend details, see [installation](installation.md) and
[backend](backend.md).

## Write the physics

The physics facade describes formulas. It lowers to model/operator IR and then
to C++. It does not run arrays in Python.

```python
from pops.physics import Model
from pops.math import ddt, div, grad, laplacian
from pops.solvers.elliptic import GeometricMG

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
flux = m.flux("F", on=U, x=[ne * E.y], y=[ne * (-E.x)], waves={"x": [E.y], "y": [-E.x]})
m.rate("explicit_rate", ddt(U) == -div(flux))
m.check()
```

## Declare the mesh and field problem

```python
import pops
from pops.fields import PoissonProblem
from pops.fields.bcs import Periodic
from pops.fields.rhs import ChargeDensity
from pops.math import laplacian
from pops.mesh.cartesian import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.solvers.elliptic import GeometricMG

mesh = CartesianMesh(n=96, L=1.0, periodic=True)
layout = Uniform(mesh)

poisson = PoissonProblem(
    name="phi",
    unknown="phi",
    equation=(-laplacian("phi") == ChargeDensity.from_blocks("ne")),
    bcs=(Periodic(),),
    solver=GeometricMG(),
)
```

## Build the time program

`pops.time` is the language. Ready schemes live in `pops.lib.time`.

```python
from pops.time import Program
from pops.lib.time import ssprk3

time = Program("advance")
ssprk3(time, "ne")
```

For a manual scheme, use version handles:

```python
from pops.numerics.terms import Flux

T = Program("manual_step")
U = T.state("U", block="ne")

fields = T.solve_fields(U.n)
R = T.rhs(state=U.n, fields=fields, terms=[Flux()])
T.define(U.next, U.n + T.dt * R)
T.commit("ne", U.next)
```

`U.n`, `U.next`, `U.stage(k)`, and `U.prev` are IR handles. They are not runtime
storage.

## Assemble, compile, bind, run

```python
from pops.codegen import Production
from pops.numerics.riemann import Rusanov
from pops.numerics.reconstruction.limiters import Minmod

case = (
    pops.Case(layout=layout, name="diocotron")
    .block("ne", physics=m, spatial=pops.FiniteVolume(limiter=Minmod(), riemann=Rusanov()))
    .field(poisson)
    .time(time)
)

compiled = pops.compile(case, backend=Production())
sim = pops.bind(compiled, state={"ne": ne0})
sim.run(t_end=0.1, cfl=0.4)
```

The generated code and runtime use the same C++ core as native runs:
finite-volume kernels, field solves, reductions, Kokkos execution, MPI
communication, and AMR infrastructure.

## Switch to AMR

AMR is a layout descriptor:

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

Use this `layout` in the same `Case`. The public flow does not change.

## Inspect

Before running, inspect what will be compiled:

```python
print(case)
case.inspect()

print(compiled)
compiled.inspect()
compiled.arguments()
```

For memory and profiling:

```python
mem = compiled.estimate_memory(grid=mesh)
print(mem)

with sim.profile(pops.Profile.Basic()) as prof:
    sim.run(t_end=0.1, cfl=0.4)

prof.summary().print()
```

Profiling is off unless requested.
