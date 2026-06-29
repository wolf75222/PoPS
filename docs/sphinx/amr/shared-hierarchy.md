# Shared hierarchy

All blocks live on a single AMR hierarchy: same boxes, same MPI distribution,
same space steps per level. This is one common hierarchy carrying several
fields, never one hierarchy per species.

- **Single-block**: one block installed on the hierarchy, with dynamic regrid
  and conservative reflux.
- **Multi-block**: several blocks co-located on the shared hierarchy. Auxiliary
  fields and elliptic solves are shared where the model declares them.

The C++ runtime verifies that all blocks share exactly the same layout per
level. That invariant is required for shared aux fields, composite field solves,
reflux, and average-down.

```python
from pops.mesh.cartesian import CartesianMesh
from pops.mesh.layouts import AMR
from pops.mesh.amr import Refine, RegridEvery, PatchLayout
from pops.codegen import Production

mesh = CartesianMesh(n=96, L=1.0, periodic=True)
layout = AMR(
    mesh,
    max_levels=2,
    ratio=2,
    regrid=RegridEvery(8),
    patches=PatchLayout(coarse_max_grid=32),
    refine=Refine.on("density").above(0.05),
)

compiled = pops.compile_problem(model=module, program=program, backend=Production(), layout=layout)
sim = pops.AmrSystem(n=96, L=1.0)
sim.install(compiled, instances={"ne": {"model": module, "initial": ne0, "spatial": spatial}},
            solvers={"phi": GeometricMG()})

sim.step_cfl(0.4)
print("fine patches:", sim.n_patches(), "| mass:", sim.mass("ne"))
rho = sim.density("ne")
```

The AMR layout descriptor carries regrid cadence, patch policy, and refinement
policy. `sim.install` receives the compiled artifact and runtime inputs; Python
does not construct patches or execute AMR kernels.
