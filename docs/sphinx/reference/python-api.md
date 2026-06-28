# Python API

This page documents the stable user-facing shape of `pops`. The implementation
also contains runtime seams used by tests and bindings; those are not the
authoring path.

For the rules, see [public API contract](public-api-contract.md).

## Main flow

```python
import pops
from pops.codegen import Production

compiled = pops.compile(case, backend=Production())
sim = pops.bind(compiled, state=initial_state, params=params)
sim.run(t_final=1.0, cfl=0.4)
```

`pops.compile` validates and lowers the `Case`. `pops.bind` attaches data and
constructs the C++ runtime from the case layout.

## Top-level assembly

```{eval-rst}
.. autoclass:: pops.Case
   :members:
```

## Runtime facade returned by bind

The object returned by `pops.bind` is the simulation facade. User code should
obtain it from `bind`, not by hand-building runtime classes.

```{eval-rst}
.. autoclass:: pops.System
   :members: run, write, checkpoint, time, mass, density, potential, profile

.. autoclass:: pops.AmrSystem
   :members: run, write, checkpoint, time, mass, density, potential, profile
```

The full runtime classes still expose lower-level methods because the C++/pybind
layer and tests use them. They are intentionally not examples for user
authoring.

## Physics authoring

```{eval-rst}
.. autoclass:: pops.physics.Model
   :members:
```

The physics facade builds model/operator IR. It does not compile directly in
the public flow; the case is compiled by `pops.compile`.

## Native brick composition

Native bricks are already-compiled C++ pieces that can be assembled into a
model. They are useful presets, but the public run flow is still `Case ->
compile -> bind -> run`.

```{eval-rst}
.. autofunction:: pops.Model

.. autoclass:: pops.Scalar
.. autoclass:: pops.FluidState
.. autoclass:: pops.ExB
.. autoclass:: pops.CompressibleFlux
.. autoclass:: pops.IsothermalFlux
.. autoclass:: pops.NoSource
.. autoclass:: pops.PotentialForce
.. autoclass:: pops.GravityForce
.. autoclass:: pops.MagneticLorentzForce
.. autoclass:: pops.PotentialMagneticForce
.. autoclass:: pops.ChargeDensity
.. autoclass:: pops.BackgroundDensity
.. autoclass:: pops.GravityCoupling
```

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
from pops.numerics.terms import Flux, SourceTerm
```

The top-level `pops.FiniteVolume(...)` function remains the current bridge to
the runtime spatial brick.

## Time

`pops.time` is the language:

```{eval-rst}
.. autoclass:: pops.time.Program
   :members:

.. autoclass:: pops.time.CompiledTime
   :members:
```

Ready schemes are in `pops.lib.time`.

## Solvers

```python
from pops.solvers.elliptic import FFT, GeometricMG
from pops.solvers.krylov import BiCGStab, CG, GMRES, Richardson
from pops.solvers.nonlinear import Newton
```

Users configure provided compiled solvers. They do not author solver loops in
Python.

## Compilation and inspection

```{eval-rst}
.. autoclass:: pops.codegen.loader.CompiledProblem
   :members: inspect, arguments, estimate_memory, dump_ir, dump_cpp, inspect_capabilities, inspect_amr

.. autoclass:: pops.codegen.loader.CompiledModel
   :members: inspect, inspect_amr
```

Useful environment variables are documented in
[environment variables](environment-variables.md).

## Profiling

```{eval-rst}
.. autoclass:: pops.Profile
   :members:

.. autoclass:: pops.PerformanceSummary
   :members:
```

Profiling is off by default and should be enabled explicitly with
`sim.profile(...)` or `POPS_PROFILE`.

## Experimental namespace

`pops.experimental` is not production API. Pages may mention it for debugging,
but tutorials must not use it as a solver route.
