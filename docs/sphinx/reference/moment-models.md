# Moment models

`pops.moments` builds moment-model physics from typed structural objects. The
package is an authoring layer; it generates PoPS model IR and C++ code. It does
not run a Python moment solver.

## Current public surface

```python
from pops.moments import CartesianVelocityMoments, RealizabilityProjection
from pops.moments.closures import HyQMOM15Closure

spec = (
    CartesianVelocityMoments(
        order=4,
        closure=HyQMOM15Closure(),
        exact_speeds=True,
        robust=True,
    )
    .add_transport()
    .add_poisson_coupling(phi="phi", eps=1.0)
    .add_vlasov_electric_source("grad_x", "grad_y", "q_over_m")
    .set_realizability(RealizabilityProjection(eps_m00=1.0e-12, eps_cov=1.0e-12))
)

model = spec.build(name="hyqmom15")
```

The `MomentModel` facade records options. `build()` is the single point that
touches the generator and returns a physics model suitable for `pops.Case`.

## Required contracts

A moment model must declare:

- the hierarchy order and ordering;
- the closure;
- realizability projection or limiter policy;
- wave-speed policy;
- source terms;
- Poisson or field coupling;
- whether Roe-style dissipation is available.

For high-order moments, positivity of the density is not sufficient. The model
must preserve the realizability cone required by the closure.

## Use in a case

```python
from pops.mesh import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.numerics.riemann import HLL
from pops.numerics.reconstruction import MUSCL
from pops.numerics.reconstruction.limiters import Minmod
from pops.time import Program
from pops.lib.time import ssprk3

spatial = pops.FiniteVolume(
    riemann=HLL(),
    reconstruction=MUSCL(limiter=Minmod()),
)

program = Program("ssprk3")
ssprk3(program, "moments")

case = (
    pops.Case(layout=Uniform(CartesianMesh(n=96, L=1.0, periodic=True)))
    .block("moments", physics=model, spatial=spatial)
    .time(program)
)
```

The same case structure applies to AMR; use `layout=AMR(...)`.
