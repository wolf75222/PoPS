# System

`pops.System` is the single-level runtime coupler: it carries the blocks, shares one Poisson
solve, and advances the whole. The documented PUBLIC way to build and run a simulation is the
typed `pops.Case` assembly lowered by `pops.compile` and wired by `pops.bind` -> `sim.run(...)`;
the per-step `System` methods (`add_block`, `add_equation`, `set_poisson`, `step_cfl`) are the
low-level seam `pops.bind` builds on and the tests use, not the front door.

## The public path: Case -> compile -> bind -> run

You author the physics with `pops.physics.Model`, declare the elliptic field with
`pops.fields.PoissonProblem`, assemble a `pops.Case`, then compile and bind a runnable simulation.

```python
import numpy as np
import pops
import pops.time as T
from pops.physics import Model
from pops.math import laplacian, grad, div, ddt
from pops.mesh.cartesian import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.fields import PoissonProblem
from pops.fields.bcs import Dirichlet
from pops.fields.rhs import ChargeDensity
from pops.solvers.elliptic import GeometricMG
from pops.mesh.geometry import Disc
from pops.codegen import Production

# Physics: a scalar density advected by the E x B drift, coupled to Poisson.
m = Model("diocotron")
U = m.state("U", components=["ne"], roles={"ne": "density"})
(ne,) = U
phi = m.field("phi")
m.solve_field("fields_from_state",
              equation=(-laplacian(phi) == ne),
              outputs={"phi": phi, "grad_x": grad(phi).x, "grad_y": grad(phi).y},
              solver="geometric_mg")
E = m.vector_field("E", x=-grad(phi).x, y=-grad(phi).y)
flux = m.flux("F", on=U, x=[ne * E.y], y=[ne * (-E.x)], waves={"x": [E.y], "y": [-E.x]})
m.rate("explicit_rate", ddt(U) == -div(flux))
m.check()

# The typed elliptic field. Use a face Dirichlet on a circular conducting wall, or Periodic().
wall = Disc(radius=0.40)
poisson = PoissonProblem(
    name="phi", unknown="phi",
    equation=(-laplacian("phi") == ChargeDensity.from_blocks("ne")),
    bcs=(Dirichlet(value=0.0, on=wall.boundary()),),
    solver=GeometricMG(),
)

case = (pops.Case(layout=Uniform(CartesianMesh(n=128, L=1.0, periodic=False)), name="diocotron")
        .block("ne", physics=m)
        .field(poisson)
        .time(T.Program("euler")))

compiled = pops.compile(case, backend=Production())
sim = pops.bind(compiled, state={"ne": ne0})        # ne0: (n, n) array
sim.run(0.05, cfl=0.4)
print("blocks:", sim.block_names())
```

`PoissonProblem(unknown=, equation=, solver=, bcs=)` is the typed elliptic surface: the unknown
field, its governing equation `-laplacian(phi) == rhs`, the solver
(`pops.solvers.elliptic.GeometricMG()` or `FFT()` for a periodic box), and the boundary
conditions. The right-hand side is a typed `ChargeDensity.from_blocks(...)`; a circular conducting
wall is a `Dirichlet` pinned to the boundary of a `pops.mesh.geometry.Disc`. `pops.compile` lowers
the assembly to a compiled handle; `pops.bind` wires a runnable `System` and routes the initial
state, parameters, aux inputs and field solvers; `sim.run(t_end, cfl=)` advances.

```{note}
**AMR variant.** Swap the layout to `pops.mesh.layouts.AMR(mesh, max_levels=2, ratio=2)` and
author the refinement with `case.amr.refine(pops.mesh.amr.Refine.on("density").above(0.5))`.
`pops.bind` then builds an `AmrSystem` from a config derived from the layout (regrid cadence,
patch layout) and flows the refinement and the Poisson field onto it before installing the block.
```

## Low-level runtime seam

`pops.bind` builds on the native `System` runtime: it calls `add_block` / `add_equation`
(install a block from a native `pops.Model(...)` brick composition or a compiled DSL model),
`set_poisson` (configure the shared elliptic), `set_density` (set state) and `step_cfl` / `step` /
`advance` (advance). These methods stay available for the native/AMR runtime and the tests, but
they are not the recommended public path. On the C++ side the coupler lives in
`runtime/system.hpp` (`System`, multi-block single-level, shared Poisson) and is exposed to Python
by `python/bindings/core/bindings.cpp`. The backend (serial / OpenMP / Kokkos GPU / MPI) is the one
with which `libadc` was compiled; the physics never sees the backend.

## Multi-block and multi-species

Several blocks co-exist in the same `System`, coupled only by the right-hand side of the
shared Poisson (`f = sum_s q_s n_s`) and, optionally, by inter-species sources; never
by the flux. Each block keeps its own model, its own spatial scheme and its own
time policy. The public assembly adds one `case.block(name, physics=...)` per species and one
`ChargeDensity.from_blocks(...)` summing the contributing blocks; multi-block lowering is wired
through the same `pops.compile` / `pops.bind` path. In the block name indexes
`sim.density(name)` / `sim.mass(name)` after bind.

Inter-species coupled sources. In addition to the coupling by the field, inter-species sources
(operator-split, applied after the transport) transfer matter, momentum or energy between blocks.
Three fixed forms exist: ionization `n_g -> n_i + n_e` (mass from the neutral to the ion),
inter-species friction (momentum conserved), and thermal exchange (energy conserved). For a
generic inter-species source described in formulas, author it on the physics model. The detail of
the multi-species / plasma case is in
[ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md) (section 18,
"composition runtime and multi-species system") and
[ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md). Exhaustive
coupling surface: [COUPLING_SURFACE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/COUPLING_SURFACE.md),
[COUPLER_HIERARCHY.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/COUPLER_HIERARCHY.md).
