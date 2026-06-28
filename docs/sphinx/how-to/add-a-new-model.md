# Add a new model

A *model* describes one equation: its flux, source, wave speeds and elliptic right-hand side.
This page covers the task of defining a model and running it in a simulation, and the choice
between the two writing fronts: native bricks and the symbolic DSL. For the concepts behind a
model, see [The physical model](../concepts/physical-model.md). For the full step-by-step
walkthrough of each front, follow the two write-a-model tutorials linked below.

This page assumes you have already built the `pops` Python module. If you have not, follow the
[installation guide](../getting-started/installation.md) first.

## Choose a writing front

Both fronts produce the same computational object on the C++ core and plug into an `pops.System`
the same way. Pick the front by whether the bricks you need already exist:

- Use **native bricks** when the bricks you need already exist. You compose four generic,
  pre-compiled bricks with `pops.Model(state, transport, source, elliptic)`. There is no
  just-in-time compilation, and you keep full production parity (MPI, AMR, GPU). Follow
  [Write a model with bricks](../tutorials/write-a-model-with-bricks.md).
- Use the **symbolic DSL** when the model you want does not exist as a native brick. You write
  the equation as formulas with `pops.physics.facade.Model`, then compile it into a `.so`. Follow
  [Write a model with the DSL](../tutorials/write-a-model-with-dsl.md).

The two fronts are interchangeable and produce an identical numerical kernel. The full brick
catalog is in the [brick reference](../reference/native-bricks.md), and the formula declarators
are in the [DSL reference](../reference/symbolic-dsl.md).

## Define the model

For native bricks, compose the four slots and let `pops.Model` validate the
state-versus-transport consistency:

```python
model = pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0), source=pops.NoSource(), elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0))
```

For the DSL, declare the conservative variables, the auxiliary fields, the flux, the eigenvalues
and the elliptic right-hand side, then call `m.check()` to verify every referenced variable is
declared, and compile:

```python
compiled = diocotron_model(n_i0).compile(backend="production")
```

Replace `production` with `aot` for a marshaled, mono-rank `.so` for CPU debug or bench. The
default backend of `m.compile` is `auto`, which auto-selects `production` under toolchain parity
with the installed `_pops`, otherwise falls back to `aot`; the explicit values
`prototype | aot | production` are still available. For the backend trade-offs, see the
[backend matrix](../reference/backend-matrix.md).

## Run the model

The documented public path assembles a typed `pops.Case`, lowers it with `pops.compile`, and binds
a runnable simulation with `pops.bind`. Declare the elliptic field, assemble the case, compile and
bind, then advance with `sim.run`:

```python
import pops.time as T
from pops.mesh.cartesian import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.fields import PoissonProblem
from pops.fields.bcs import Periodic
from pops.fields.rhs import ChargeDensity
from pops.solvers.elliptic import GeometricMG
from pops.codegen import Production
from pops.math import laplacian

poisson = PoissonProblem(name="phi", unknown="phi",
                         equation=(-laplacian("phi") == ChargeDensity.from_blocks("ne")),
                         bcs=(Periodic(),), solver=GeometricMG())

case = (pops.Case(layout=Uniform(CartesianMesh(n=96, L=1.0, periodic=True)))
        .block("ne", physics=m)              # m: the pops.physics.Model authored above
        .field(poisson)
        .time(T.Program("euler")))

compiled = pops.compile(case, backend=Production())
sim = pops.bind(compiled, state={"ne": ne0})  # ne0: contiguous 2D array, indexed ne[j, i]
sim.run(0.1, cfl=0.4)
```

A native `pops.Model(...)` brick composition (the `ModelSpec` above) is not compiled through
`pops.compile`; it plugs into the low-level native runtime directly with
`sim.add_block("ne", model=model, ...)` / `sim.set_poisson(...)` / `sim.set_density(...)` /
`sim.step_cfl(...)`. Those runtime methods stay for the native/AMR runtime and the tests; they are
not the documented front door.

## Next steps

- [Write a model with bricks](../tutorials/write-a-model-with-bricks.md) for the native front.
- [Write a model with the DSL](../tutorials/write-a-model-with-dsl.md) for the formula front.
- [Models overview](../models/index.md) for the hybrid front (`pops.CompositeModel`) and the
  `PhysicalModel` contract.
- [Configure a simulation](../simulation/index.md) to wire the system, Poisson and time integration.
