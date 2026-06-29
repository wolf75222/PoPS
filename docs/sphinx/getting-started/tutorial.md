# Tutorial

This tutorial follows the public route:

```text
physics/model authoring
    -> pops.model.Module
    -> pops.time.Program
    -> pops.compile_problem(...)
    -> pops.System(...)
    -> sim.install(compiled, ...)
    -> sim.step_cfl(...)
```

Python describes typed objects. C++/Kokkos/MPI executes.

## 1. Build a model

```python
from pops.physics import Model
from pops.math import ddt, div

m = Model("scalar_transport")
U = m.state("U", components=["rho"], roles={"rho": "density"})
(rho,) = U
flux = m.flux("F", on=U, x=[rho], y=[0.0 * rho], waves={"x": [1.0 + 0.0 * rho]})
m.rate("explicit_rate", ddt(U) == -div(flux))

module = m.to_module()
```

## 2. Build a time program

```python
from pops.time import Program
from pops.lib.time import ssprk3

program = Program("advance").bind_operators(module)
ssprk3(program, "plasma", rhs_operator=module.operator_registry().get("explicit_rate"))
```

## 3. Compile and install

```python
import pops
from pops.codegen import Production
from pops.mesh.cartesian import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.numerics.riemann import Rusanov
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.spatial import spatial

mesh = CartesianMesh(n=96, L=1.0, periodic=True)
layout = Uniform(mesh)
fv = spatial.FiniteVolume(reconstruction=Minmod(), riemann=Rusanov())

compiled = pops.compile_problem(model=module, time=program, backend=Production(), layout=layout)

sim = pops.System(n=96, L=1.0, periodic=True)
sim.install(compiled, instances={"plasma": {"model": module, "initial": U0, "spatial": fv}})
```

## 4. Advance

```python
t_final = 0.1
cfl = 0.4

while sim.time() < t_final:
    sim.step_cfl(cfl)
```

Use [public API contract](../reference/public-api-contract.md) as the source of
truth for what is public.
