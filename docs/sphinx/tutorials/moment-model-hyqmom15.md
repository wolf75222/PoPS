# Build a HyQMOM15 moment model

This tutorial shows the current PoPS moment authoring flow. Python declares the
hierarchy, closure, realizability projection, and numerical descriptors. The
generated/native C++ route performs the update.

## Declare the hierarchy

```python
from pops.moments import CartesianVelocityMoments, RealizabilityProjection
from pops.moments.closures import HyQMOM15Closure

moment_spec = (
    CartesianVelocityMoments(order=4, closure=HyQMOM15Closure(), exact_speeds=True)
    .add_transport()
    .add_poisson_coupling(phi="phi", eps=1.0)
    .add_vlasov_electric_source("grad_x", "grad_y", "q_over_m")
    .set_realizability(
        RealizabilityProjection(eps_m00=1.0e-12, eps_cov=1.0e-12, robust=True)
    )
)

print(moment_spec.hierarchy())
model = moment_spec.build(name="hyqmom15")
```

`order=4` gives a 15-moment hierarchy in 2D. The closure supplies the next-order
moments required by the transport. The realizability projection protects the
closure domain; it is stronger than a density-only positivity check.

## Select finite-volume numerics

```python
from pops.numerics.riemann import HLL
from pops.numerics.reconstruction import MUSCL
from pops.numerics.reconstruction.limiters import Minmod

spatial = pops.FiniteVolume(
    riemann=HLL(),
    reconstruction=MUSCL(limiter=Minmod()),
)
```

Use HLL/Rusanov first for robustness. Roe-like routes require a moment closure
that declares the required characteristic structure.

## Assemble and run

```python
from pops.mesh import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.time import Program
from pops.lib.time import ssprk3
from pops.codegen import Production

program = Program("ssprk3")
ssprk3(program, "moments")

case = (
    pops.Case(layout=Uniform(CartesianMesh(n=64, L=1.0, periodic=True)))
    .block("moments", physics=model, spatial=spatial)
    .time(program)
)

compiled = pops.compile(case, backend=Production())
print(compiled.arguments())

sim = pops.bind(compiled, state={"moments": M0}, params={"q_over_m": -1.0})
sim.run(t_end=0.1, cfl=0.4)
```

## AMR

Switching to AMR is a layout change:

```python
from pops.mesh.layouts import AMR
from pops.mesh.amr import Refine, RegridEvery

layout = AMR(
    base=CartesianMesh(n=64, L=1.0, periodic=True),
    max_levels=2,
    ratio=2,
    regrid=RegridEvery(4),
    refine=Refine.on("density").above(0.05),
)
```

The moment model, spatial descriptors, and time program stay the same.
