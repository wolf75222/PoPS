# Python API

This page documents the stable user-facing shape of `pops`. The full rule set
is in [public API contract](public-api-contract.md).

## Main flow

```python
import pops
from pops.codegen import Production

compiled = pops.compile_problem(model=module, time=program, backend=Production(), layout=layout)

sim = pops.System(n=mesh.n, L=mesh.L, periodic=mesh.periodic)
sim.install(
    compiled,
    instances={"plasma": {"model": module, "initial": U0, "spatial": spatial}},
    params=params,
    aux=aux,
    solvers=solvers,
)
sim.step_cfl(0.4)
```

`compile_problem` validates and lowers the model/program pair into one compiled
problem artifact. `System.install` or `AmrSystem.install` attaches runtime data
and installs that artifact. Numerical execution is C++/Kokkos/MPI work.

## Compile artifact

```{eval-rst}
.. autofunction:: pops.compile_problem

.. autoclass:: pops.CompiledProblem
   :members: inspect, arguments, estimate_memory, dump_ir, dump_cpp
```

## Runtime facade

User code constructs the runtime facade explicitly, then installs the compiled
problem. Low-level wiring methods are private implementation seams.

```{eval-rst}
.. autoclass:: pops.System
   :members: install, step_cfl, step, write, checkpoint, time, mass, density, potential, profile

.. autoclass:: pops.AmrSystem
   :members: install, step_cfl, step, write, checkpoint, time, mass, density, potential, profile
```

## Physics authoring

```{eval-rst}
.. autoclass:: pops.physics.Model
   :members:
```

The physics facade builds model/operator IR. It does not compile directly and
does not own runtime data.

## Mesh and layouts

```{eval-rst}
.. autoclass:: pops.mesh.cartesian.CartesianMesh
.. autoclass:: pops.mesh.layouts.Uniform
.. autoclass:: pops.mesh.layouts.AMR
```

AMR policies live under `pops.mesh.amr`.

## Fields

```{eval-rst}
.. autoclass:: pops.fields.FieldProblem
.. autoclass:: pops.fields.PoissonProblem
.. autoclass:: pops.fields.ScreenedPoissonProblem
.. autoclass:: pops.fields.AnisotropicPoissonProblem
```

Boundary conditions, right-hand-side descriptors, coefficients, nullspaces, and
field cadence policies live in `pops.fields.bcs`, `pops.fields.rhs`,
`pops.fields.coefficients`, `pops.fields.nullspace`, and `pops.fields.policies`.

## Numerics

Numerical choices are descriptors.

```python
from pops.numerics.riemann import HLL, HLLC, Roe, Rusanov
from pops.numerics.reconstruction import MUSCL, WENO5Z
from pops.numerics.reconstruction.limiters import Minmod, VanLeer
from pops.numerics.spatial import spatial
from pops.numerics.terms import Flux, SourceTerm

fv = spatial.FiniteVolume(reconstruction=Minmod(), riemann=Rusanov())
```

## Time

`pops.time` is the language:

```{eval-rst}
.. autoclass:: pops.time.Program
   :members: state, call, define, commit, history, store_history
```

Ready schemes live in `pops.lib.time`, not in `pops.time`.

## Runtime profiling

```python
from pops.runtime import Profile

with sim.profile(Profile.Advanced()) as prof:
    sim.step_cfl(0.4)

prof.summary().print()
```
