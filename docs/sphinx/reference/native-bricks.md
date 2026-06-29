# Native bricks

Native bricks are provided C++ pieces of physics and numerics. Python selects
and composes them as typed objects; computation stays in C++.

## Physics bricks

Native model presets should lower to `pops.model.Module` objects that can enter
the compiled problem route:

```python
module = native_model.to_module()
compiled = pops.compile_problem(model=module, time=program, backend=Production(), layout=layout)
sim = pops.System(n=mesh.n, L=mesh.L, periodic=mesh.periodic)
sim.install(compiled, instances={"ne": {"model": module, "initial": ne0, "spatial": spatial}})
```

## Spatial brick

Finite-volume spatial choices live under `pops.numerics`:

```python
from pops.numerics.riemann import Rusanov
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.spatial import spatial

fv = spatial.FiniteVolume(reconstruction=Minmod(), riemann=Rusanov())
```

The physical flux of a model and the numerical Riemann flux are different
concepts. The former is declared by the model; the latter is selected by the
spatial descriptor.

## Field and solver bricks

Field solves are described through model/field descriptors and solver
descriptors:

```python
from pops.solvers.elliptic import GeometricMG

sim.install(compiled, instances={"ne": {"model": module, "initial": ne0}},
            solvers={"phi": GeometricMG()})
```

Solver descriptors live in `pops.solvers`, not in `pops.lib`.

## Time

Ready time schemes are macros in `pops.lib.time`:

```python
from pops.time import Program
from pops.lib.time import ssprk3

program = Program("advance")
ssprk3(program, "ne", rhs_operator=module.operator_registry().get("explicit_rate"))
```
