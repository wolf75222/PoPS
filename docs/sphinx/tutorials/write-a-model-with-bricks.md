# Write a model with ready-made bricks

Use this tutorial when the physics you need is already available as compiled
PoPS bricks or presets. The Python code selects typed descriptors and assembles a
case; the C++ runtime executes the loops.

## Build the pieces

```python
from pops.mesh import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.numerics.riemann import Rusanov
from pops.numerics.reconstruction import MUSCL
from pops.numerics.reconstruction.limiters import Minmod
from pops.time import Program
from pops.lib.time import ssprk3
from pops.codegen import Production

mesh = CartesianMesh(n=128, L=1.0, periodic=True)
layout = Uniform(mesh)

spatial = pops.FiniteVolume(
    riemann=Rusanov(),
    reconstruction=MUSCL(limiter=Minmod()),
)
```

Use a ready-made physics model from `pops.lib.models` when one exists:

```python
from pops.lib.models import diocotron

model = diocotron.scalar_exb(background_density=n_i0)
```

The exact model preset depends on the library catalog. If no preset matches,
write the physics with `pops.physics.Model` and lower it to `pops.model`.

## Add a field solve

```python
from pops.fields import PoissonProblem
from pops.fields.bcs import Periodic
from pops.fields.rhs import ChargeDensity
from pops.math import laplacian
from pops.solvers.elliptic import GeometricMG

phi = "phi"
poisson = PoissonProblem(
    name="phi",
    unknown=phi,
    equation=(-laplacian(phi) == ChargeDensity.from_blocks("electrons")),
    bcs=(Periodic(),),
    solver=GeometricMG(),
)
```

## Assemble and run

```python
program = Program("ssprk3")
ssprk3(program, "electrons")

case = (
    pops.Case(layout=layout, name="diocotron")
    .block("electrons", physics=model, spatial=spatial)
    .field(poisson)
    .time(program)
)

compiled = pops.compile(case, backend=Production())
sim = pops.bind(compiled, state={"electrons": ne0})
sim.run(t_final=0.1, cfl=0.4)
```

## Switch to AMR

```python
from pops.mesh.layouts import AMR
from pops.mesh.amr import Refine, RegridEvery

layout = AMR(
    base=mesh,
    max_levels=2,
    ratio=2,
    regrid=RegridEvery(4),
    refine=Refine.on("density").above(0.05),
)
```

Keep the same model, field problem, spatial descriptors, and time program. The
layout descriptor selects the AMR C++ route during compile and bind.
