# Spectral FFT Poisson

Use FFT Poisson for uniform periodic meshes. It is a solver descriptor, not a
runtime string selector.

```python
from pops.fields import PoissonProblem
from pops.fields.bcs import Periodic
from pops.fields.rhs import ChargeDensity
from pops.math import laplacian
from pops.solvers.elliptic import FFT

poisson = PoissonProblem(
    name="phi",
    unknown="phi",
    equation=(-laplacian("phi") == ChargeDensity.from_blocks("plasma")),
    bcs=(Periodic(),),
    solver=FFT(),
)
```

FFT requires a uniform periodic layout:

```python
from pops.mesh import CartesianMesh
from pops.mesh.layouts import Uniform

layout = Uniform(CartesianMesh(n=256, L=1.0, periodic=True))
```

For AMR or non-periodic physical boundaries, choose a compatible solver such as
`GeometricMG()`.
