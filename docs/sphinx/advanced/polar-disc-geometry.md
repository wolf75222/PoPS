# Polar and embedded geometry

Geometry belongs to the mesh layer. Numerical methods stay descriptors for
discretization, reconstruction, and Riemann solves.

```python
from pops.mesh import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.mesh.geometry import Disc

wall = Disc(center=(0.0, 0.0), radius=0.4)
mesh = CartesianMesh(n=256, L=1.0, periodic=False)
layout = Uniform(mesh, embedded_boundary=wall)
```

Use the layout in a case:

```python
case = pops.Case(layout=layout).block("plasma", physics=model, spatial=spatial)
```

Geometry descriptors must declare their compatibility with AMR, MPI, GPU, and
field solvers. If a geometry route needs cut cells, masks, or level sets, expose
those as typed descriptors under `pops.mesh`, not as ad hoc runtime flags.
