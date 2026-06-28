# Native bricks

Native bricks are provided C++ pieces of physics and numerics. Python selects
and composes them as typed objects; the computation stays in C++.

Use them inside a `Case`. Do not document low-level runtime assembly as the
front door.

## Physics bricks

`pops.Model(state=, transport=, source=, elliptic=)` composes four native
physics roles and returns a model object that can be attached to a `Case`.

| Role | Examples |
| --- | --- |
| State | `Scalar()`, `FluidState.compressible(...)`, `FluidState.isothermal(...)` |
| Transport | `ExB(...)`, `CompressibleFlux()`, `IsothermalFlux()` |
| Source | `NoSource()`, `PotentialForce(...)`, `GravityForce()`, `MagneticLorentzForce(...)`, `PotentialMagneticForce(...)` |
| Elliptic RHS | `ChargeDensity(...)`, `BackgroundDensity(...)`, `GravityCoupling(...)` |

Example:

```python
import pops

model = pops.Model(
    state=pops.Scalar(),
    transport=pops.ExB(B0=1.0),
    source=pops.NoSource(),
    elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0),
)
```

Then attach it to a case:

```python
case = pops.Case(layout=layout, name="diocotron").block("ne", physics=model)
```

## Spatial brick

The current bridge to the runtime finite-volume brick is `pops.FiniteVolume`.
It accepts typed numerics descriptors.

```python
from pops.numerics.riemann import Rusanov
from pops.numerics.reconstruction.limiters import Minmod

spatial = pops.FiniteVolume(limiter=Minmod(), riemann=Rusanov())
case = case.block("ne", physics=model, spatial=spatial)
```

The physical flux of a model and the numerical Riemann flux are different
concepts. The former is declared by the model; the latter is selected by the
spatial descriptor.

## Field and solver bricks

Field solves are described through `pops.fields` and solver descriptors:

```python
from pops.fields import PoissonProblem
from pops.fields.bcs import Periodic
from pops.fields.rhs import ChargeDensity
from pops.math import laplacian
from pops.solvers.elliptic import GeometricMG

poisson = PoissonProblem(
    name="phi",
    unknown="phi",
    equation=(-laplacian("phi") == ChargeDensity.from_blocks("ne")),
    bcs=(Periodic(),),
    solver=GeometricMG(),
)

case = case.field(poisson)
```

Solver descriptors live in `pops.solvers`, not `pops.lib`.

## Time

Ready time schemes are macros in `pops.lib.time`:

```python
from pops.time import Program
from pops.lib.time import ssprk3

time = Program("advance")
ssprk3(time, "ne")
case = case.time(time)
```

`pops.time` is the language. `pops.lib.time` contains ready schemes.

## Inter-species couplings

Coupling descriptors describe source exchanges between named blocks. They must
lower to C++ kernels through the case/program route.

```python
collision = pops.Collision("ions", "electrons", rate=nu)
```

When a coupling is part of a public workflow, it must be represented in the
case/program and validated before run.

## Experimental and debug tools

`pops.experimental.PythonFlux` exists for debugging and tests. It computes on
the host in Python and must not be used in production tutorials.

## Inspecting bricks

Descriptors should answer:

```python
brick.inspect()
brick.requirements()
brick.capabilities()
brick.options()
```

Validation should fail before runtime when a selected brick requires a model
capability that the model does not declare.
