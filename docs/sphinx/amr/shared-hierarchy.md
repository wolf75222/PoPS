# Shared hierarchy


All blocks live on a single AMR hierarchy: same boxes, same MPI distribution
(`DistributionMapping`), same space steps per level. This is the model "a common hierarchy
carrying several fields", never one hierarchy per species. The current version
carries two levels (refinement ratio 2: the fine level has a step `dx/2`).

- **Single-block** (a single `add_block`): historical path `AmrCouplerMP`, with dynamic regrid
  and conservative reflux. Bit-identical to what it has always produced.
- **Multi-block** (two or more `add_block`): N blocks co-located on the shared
  hierarchy (engine `AmrRuntime`). A single auxiliary channel per level (`phi`, `grad phi`)
  and a single coarse Poisson whose right-hand side is the co-located sum of the elliptic
  bricks of the blocks (`f = somme_b q_b n_b`, read at the same cells). Conservation is
  ensured per block (reflux + average-down). In multi-block, the block name indexes
  `set_density(name)`, `mass(name)` and `density(name)`.

A guard (`same_layout_or_throw`) verifies at construction that all blocks share
exactly the same layout per level (boxes, order, distribution, `dx`/`dy`): this is the
precondition of the single aux and the single Poisson. Detail: [ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md)
section 8, [AMR_MULTIBLOCK_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_MULTIBLOCK_DESIGN.md) sections 1-2, and the core
`include/pops/runtime/amr_system.hpp`.

```python
import numpy as np
import pops
import pops.time as T
from pops.physics import Model
from pops.math import laplacian, grad, div, ddt
from pops.mesh.cartesian import CartesianMesh
from pops.mesh.layouts import AMR
from pops.mesh.amr import Refine, RegridEvery
from pops.fields import PoissonProblem
from pops.fields.bcs import Periodic
from pops.fields.rhs import ChargeDensity
from pops.solvers.elliptic import GeometricMG
from pops.codegen import Production

n, L = 96, 1.0
ne0 = np.ones((n, n))                 # initial density (n, n), row-major

m = Model("diocotron")
U = m.state("U", components=["ne"], roles={"ne": "density"})
(ne,) = U
phi = m.field("phi")
m.solve_field("fields_from_state", equation=(-laplacian(phi) == ne),
              outputs={"phi": phi, "grad_x": grad(phi).x, "grad_y": grad(phi).y},
              solver="geometric_mg")
E = m.vector_field("E", x=-grad(phi).x, y=-grad(phi).y)
flux = m.flux("F", on=U, x=[ne * E.y], y=[ne * (-E.x)], waves={"x": [E.y], "y": [-E.x]})
m.rate("explicit_rate", ddt(U) == -div(flux))
m.check()

poisson = PoissonProblem(name="phi", unknown="phi",
                         equation=(-laplacian("phi") == ChargeDensity.from_blocks("ne")),
                         bcs=(Periodic(),), solver=GeometricMG())

layout = AMR(CartesianMesh(n=n, L=L, periodic=True), max_levels=2, ratio=2, regrid=RegridEvery(8))
case = (pops.Case(layout=layout).block("ne", physics=m).field(poisson).time(T.Program("euler")))
case.amr.refine(Refine.on("density").above(0.05))   # refine where the density exceeds the threshold

compiled = pops.compile(case, backend=Production())
sim = pops.bind(compiled, state={"ne": ne0})
sim.run(0.25, cfl=0.4)                # CFL on the coarse-level step

print("fine patches:", sim.n_patches(), "| mass:", sim.mass("ne"))
rho = sim.density("ne")               # coarse density (n, n)
```

`pops.bind` builds the `AmrSystem` from a config derived from the `AMR` layout (regrid cadence from
`RegridEvery(...)`, patch settings from a `PatchLayout`) and flows the typed refinement
(`case.amr.refine(...)`) and the Poisson field onto it before installing the block.
