# AMR

AMR is a mesh layout. It is not a separate user API and not a compile target.

The public flow is identical to a uniform run:

```python
case = pops.Case(layout=layout, name="run")
compiled = pops.compile(case, backend=Production())
sim = pops.bind(compiled, state=state)
sim.run(t_end=1.0, cfl=0.4)
```

Switching from uniform to AMR changes only the layout descriptor and the AMR
policies attached to it.

```python
from pops.mesh.cartesian import CartesianMesh
from pops.mesh.layouts import AMR
from pops.mesh.amr import PatchLayout, Refine, RegridEvery

mesh = CartesianMesh(n=128, L=1.0, periodic=True)
layout = AMR(
    mesh,
    max_levels=2,
    ratio=2,
    regrid=RegridEvery(8),
    patches=PatchLayout(coarse_max_grid=32),
    refine=Refine.on("density").above(0.05),
)
```

`pops.compile` derives the AMR runtime route from `layout=AMR(...)`. User
documentation must not ask users to instantiate the AMR runtime directly.

```{toctree}
:maxdepth: 1

shared-hierarchy
tagging-regrid
prolongation-restriction
reflux
multi-block-amr
current-limits
```

## Compatibility rule

If a feature is public on `Case`, it must have a complete AMR route unless its
descriptor declares a precise mathematical incompatibility. Missing Python
plumbing, missing codegen, or missing runtime binding is an implementation bug,
not a documented limitation.

AMR features must be validated before runtime:

- layout level count and ratio;
- refinement subjects;
- field solver compatibility;
- output/checkpoint policy compatibility;
- solver/backend/platform capabilities;
- halo and reconstruction requirements;
- MPI/GPU support.

## What AMR owns

AMR owns hierarchy construction, patch layout, tagging, regrid cadence, proper
nesting, prolongation/restriction, reflux, level-aware field/output policies,
and the C++ runtime route. It does not own physics formulas.
