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
touches the generator and returns a model that must lower to `pops.model.Module`.

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

## Compile and install

```python
from pops.mesh import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.numerics.riemann import HLL
from pops.numerics.spatial import spatial
from pops.numerics.reconstruction import MUSCL
from pops.numerics.reconstruction.limiters import Minmod
from pops.time import Program
from pops.lib.time import ssprk3

fv = spatial.FiniteVolume(
    riemann=HLL(),
    reconstruction=MUSCL(limiter=Minmod()),
)

module = model.to_module()
program = Program("ssprk3")
ssprk3(program, "moments", rhs_operator=module.operator_registry().get("explicit_rate"))

mesh = CartesianMesh(n=96, L=1.0, periodic=True)
layout = Uniform(mesh)
compiled = pops.compile_problem(model=module, program=program, layout=layout)
sim = pops.System(n=96, L=1.0, periodic=True)
sim.install(compiled, instances={"moments": {"model": module, "initial": M0, "spatial": fv}})
```

The same compiled-problem structure applies to AMR; use `layout=AMR(...)` and
install on `pops.AmrSystem`.
