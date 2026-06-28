# Add or use a native brick

A native brick is a C++ implementation selected by a Python descriptor. It is
not a Python solver loop. Use native bricks when the required state, flux,
source, field RHS, reconstruction, or solver already exists in the compiled
core.

## Use existing bricks

```python
from pops.numerics.riemann import HLL
from pops.numerics.reconstruction import MUSCL
from pops.numerics.reconstruction.limiters import Minmod

spatial = pops.FiniteVolume(
    riemann=HLL(),
    reconstruction=MUSCL(limiter=Minmod()),
)
```

If a library model exists, import it from `pops.lib.models`. `pops.lib` is for
ready-made models and presets; core authoring primitives live in `pops.physics`,
`pops.numerics`, `pops.fields`, `pops.mesh`, `pops.time`, and `pops.solvers`.

## Add a new C++ brick

1. Implement the C++ brick under `include/pops/...`.
2. Bind the native ID in the pybind layer.
3. Add a Python descriptor in the owning package.
4. Declare requirements, capabilities, options, and validation.
5. Add tests that compile or bind the descriptor through a `pops.Case`.

The descriptor is the public API:

```python
spatial = pops.FiniteVolume(
    riemann=MyNativeRiemann(option=value),
    reconstruction=MUSCL(limiter=Minmod()),
)
```

Do not expose a public Python hook that computes fluxes per cell. If a debug
prototype is useful, place it under `pops.experimental` and keep it out of the
production docs.

## Verify the route

```python
compiled = pops.compile(case, backend=Production())
compiled.inspect()
compiled.dump_cpp()

sim = pops.bind(compiled, state=state)
sim.run(t_end=0.1, cfl=0.4)
```

The test should cover uniform layout and, when the brick declares AMR support,
`layout=AMR(...)`.
