# Poisson solvers


The elliptic stage solves `lap(phi) = f` (or a generalization) at each step, and it is
the core of the coupling: `f` depends on the density, and `phi` (via `grad phi`) drives the
drift. You declare it with a typed `pops.fields.PoissonProblem` and attach it to the case with
`case.field(...)`:

```python
import pops
from pops.fields import PoissonProblem
from pops.fields.bcs import Periodic, Dirichlet
from pops.fields.rhs import ChargeDensity
from pops.solvers.elliptic import GeometricMG, FFT
from pops.math import laplacian

poisson = PoissonProblem(
    name="phi", unknown="phi",
    equation=(-laplacian("phi") == ChargeDensity.from_blocks("ne")),
    bcs=(Periodic(),),                # or (Dirichlet(),) for a fixed boundary value
    solver=GeometricMG(),             # or FFT() / FFT(spectral=True) for a periodic box
)
# case.field(poisson) -> pops.compile(case, backend=Production()) -> pops.bind(...)
```

The solver is a typed object: `pops.solvers.elliptic.GeometricMG()` (multigrid, any boundary) or
`FFT()` (periodic box). The right-hand side is a typed `ChargeDensity.from_blocks(...)` (`q n`,
summed over the contributing blocks). The boundary conditions are typed `Periodic()` / `Dirichlet()`
/ `Neumann()` (a face Dirichlet on a circular conducting wall is
`Dirichlet(value=0.0, on=pops.mesh.geometry.Disc(radius=R).boundary())`).

```{note}
The low-level `sim.set_poisson(rhs=..., solver=..., bc=...)` runtime method that `pops.bind` calls
internally is the seam this typed surface lowers onto; it stays for the native/AMR runtime and the
tests, not as the documented front door.
```

## Going further


- Elliptic algorithms (multigrid, FFT, eps/Helmholtz/anisotropic, cut-cell):
  [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md), sections 9 to 12.
- The headers: `include/pops/numerics/elliptic/mg/geometric_mg.hpp`,
  `poisson_fft_solver.hpp`, `poisson_operator.hpp`.
- Conservation properties of the coupled scheme (exact FV mass, momentum, energy, values
  measured by the tests): [CONSERVATION_SUMMARY.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/CONSERVATION_SUMMARY.md).
