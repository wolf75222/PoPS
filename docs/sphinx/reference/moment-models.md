# Moment models

`pops.moments` builds moment-model physics from typed structural objects. The
package is an authoring layer; it generates PoPS model IR and C++ code. It does
not run a Python moment solver.

## Current public surface

```python
from pops.moments import (
    CartesianVelocityMoments,
    RealizabilityProjection,
    VlasovElectricSource,
)
from pops.moments.closures import gaussian_closure

spec = (
    CartesianVelocityMoments(
        order=2,
        closure=gaussian_closure(2),
        exact_speeds=True,
        robust=True,
    )
    .add_transport()
    .add_poisson_coupling(phi="phi", eps=1.0)
    .add_source(VlasovElectricSource(electric_field=("grad_x", "grad_y")))
    .set_realizability(RealizabilityProjection(eps_m00=1.0e-12, eps_cov=1.0e-12))
)

module = spec.to_module(name="custom_moments")
```

The `MomentModel` facade records options. `to_module()` is the public route used
by `compile_problem`; `build()` remains the single implementation point that
touches the moment generator.

For the provided order-4 HyQMOM15 model, use the ready model package:

```python
from pops.lib.models.moments import HyQMOM15

module = HyQMOM15.vlasov_poisson_magnetic(
    robust=True,
    exact_speeds=False,
).to_module()
```

The generic `CartesianVelocityMoments(order=4, exact_speeds=True)` route is
intentionally rejected until an order-4 exact wave-speed descriptor exists. Use
`exact_speeds=False` or a provided model-specific route.

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
import pops
from pops.mesh import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.numerics.riemann import Rusanov
from pops.numerics.spatial import FiniteVolume
from pops.numerics.reconstruction import MUSCL
from pops.numerics.reconstruction.limiters import Minmod
from pops.time import Program

fv = FiniteVolume(
    riemann=Rusanov(),
    reconstruction=MUSCL(limiter=Minmod()),
)

program = Program("moments_forward_euler")
program.bind_operators(module)
U = program.state("U", block="moments", space=module.state_spaces()["U"])
fields_op = module.operator_registry().get("fields_from_state")
rate_op = module.operator_registry().get("explicit_rate")
fields_n = program.call(fields_op, U.n, name="fields_n")
R_n = program.call(rate_op, U.n, fields_n, name="R_n")
program.define(U.next, U.n + program.dt * R_n)
program.commit(U.next, fields=fields_n)

mesh = CartesianMesh(n=96, L=1.0, periodic=True)
layout = Uniform(mesh)
compiled = pops.compile_problem(model=module, program=program, layout=layout)
sim = pops.System(layout=layout)
sim.install(compiled, instances={"moments": {"initial": M0, "spatial": fv}})
```

The same compiled-problem structure applies to AMR; use `layout=AMR(...)` and
construct the runtime with `pops.System(layout=layout)`.
