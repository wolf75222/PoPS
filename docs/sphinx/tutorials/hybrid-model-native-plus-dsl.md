# Combine native bricks and authored physics

Use native descriptors for the parts PoPS already provides, and author only the
missing physics with `pops.physics.Model`. The result still lowers through one
`pops.Case` and runs in C++.

## Author the missing operator

```python
from pops.physics import Model
from pops.math import div, ddt

m = Model("custom_source_model")
U = m.state("U", components=["rho"], roles={"rho": "density"})
rho = U[0]

F = m.flux("transport", on=U, x=[rho], y=[0.0 * rho], waves={"x": [1.0], "y": [0.0]})
S = m.source("reaction", on=U, value=[-0.1 * rho])
m.rate("explicit_rate", ddt(U) == -div(F) + S)

model = m.lower()
```

## Reuse native numerics

```python
from pops.numerics.riemann import HLL
from pops.numerics.reconstruction import MUSCL
from pops.numerics.reconstruction.limiters import Minmod

spatial = pops.FiniteVolume(
    riemann=HLL(),
    reconstruction=MUSCL(limiter=Minmod()),
)
```

## Assemble the case

```python
from pops.mesh import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.time import Program
from pops.lib.time import ssprk3
from pops.codegen import Production

program = Program("ssprk3")
ssprk3(program, "plasma")

case = (
    pops.Case(layout=Uniform(CartesianMesh(n=96, L=1.0, periodic=True)))
    .block("plasma", physics=model, spatial=spatial)
    .time(program)
)

compiled = pops.compile(case, backend=Production())
sim = pops.bind(compiled, state={"plasma": U0})
sim.run(t_end=0.1, cfl=0.4)
```

This pattern keeps the boundary clean: Python authors descriptors and IR; C++
executes the generated/native route.
