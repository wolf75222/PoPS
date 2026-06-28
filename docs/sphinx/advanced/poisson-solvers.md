# Poisson and elliptic solvers

Declare field solves with typed field problems. The problem describes the
operator, right-hand side, boundary conditions, outputs, and compiled solver
descriptor. The runtime materializes the solve.

## Geometric multigrid

```python
from pops.fields import PoissonProblem
from pops.fields.bcs import Periodic
from pops.fields.rhs import ChargeDensity
from pops.math import laplacian
from pops.solvers.elliptic import GeometricMG
from pops.solvers.tolerances import Relative

poisson = PoissonProblem(
    name="phi",
    unknown="phi",
    equation=(-laplacian("phi") == ChargeDensity.from_blocks("plasma")),
    bcs=(Periodic(),),
    solver=GeometricMG(tolerance=Relative(1.0e-10)),
)
```

Add it to a case:

```python
case = case.field(poisson)
```

## Solver choice

Use `pops.solvers` descriptors for algorithms:

- `GeometricMG()` for geometric multigrid;
- `FFT()` for periodic uniform Poisson routes;
- `CG()`, `GMRES()`, or `BiCGStab()` for matrix-free linear problems where the
  route supports Krylov solves.

The descriptor declares compatibility. For example, FFT is a uniform periodic
solver, while AMR field solves should route to multigrid.

## Multiple named fields

Declare each field as a separate `FieldProblem`:

```python
phi = PoissonProblem(name="phi", unknown="phi", equation=eq_phi, solver=GeometricMG())
psi = PoissonProblem(name="psi", unknown="psi", equation=eq_psi, solver=GeometricMG())

case = case.field(phi).field(psi)
```

The names are stable user identifiers. Solver behavior remains typed.
