# Models

A model describes the local physics of a block: state variables, physical flux,
source terms, field coupling, wave speeds, projections, and capabilities. It
does not describe mesh layout, AMR, MPI, Kokkos, output, or time stepping.

The documented route is to attach models to a `Case`.

## Ways to author a model

| Route | Use when |
| --- | --- |
| Native bricks | The physics is already provided as compiled C++ bricks. |
| `pops.physics.Model` | You need to write formulas in the Python DSL and lower them to C++. |
| Moment tools | You build a moment hierarchy from generic moment objects and closures. |
| Presets under `pops.lib.models` | You want a provided model assembly. |

All routes must produce compiled C++ execution. Python is an authoring surface.

## Native bricks

```python
import pops

model = pops.Model(
    state=pops.Scalar(),
    transport=pops.ExB(B0=1.0),
    source=pops.NoSource(),
    elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0),
)
```

## Physics DSL

```python
from pops.physics import Model
from pops.math import ddt, div

m = Model("plasma")
U = m.state("U", components=["rho", "mx", "my"], roles={"rho": "density"})
rho, mx, my = U
flux = m.flux("F", on=U, x=[...], y=[...], waves={"x": [...], "y": [...]})
m.rate("explicit_rate", ddt(U) == -div(flux))
```

The model is lowered to a module and compiled with a time program:

```python
module = m.to_module()
compiled = pops.compile_problem(model=module, time=program, backend=Production(), layout=layout)
sim = pops.System(n=mesh.n, L=mesh.L, periodic=mesh.periodic)
sim.install(compiled, instances={"plasma": {"model": module, "initial": U0, "spatial": spatial}})
```

## Moment models

Generic moment construction belongs in `pops.moments`. Ready moment models
belong in `pops.lib.models.moments`.

Moment models must declare more than positivity of density. They need explicit
contracts for realizability, closure, projection, wave speeds, moment ordering,
and compatibility with Riemann solvers.

## What a model must not do

A model must not compile itself, allocate runtime storage, run a simulation,
choose MPI/Kokkos backends, or own AMR. Those concerns belong to
`pops.compile_problem`, `sim.install`, and the C++ runtime.
