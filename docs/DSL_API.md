# PoPS public DSL API

This page is the root-level summary of the public Python DSL. The detailed
reference lives in the Sphinx docs:

- `docs/sphinx/reference/public-api-contract.md`
- `docs/sphinx/reference/python-api.md`
- `docs/sphinx/reference/time-program.md`
- `docs/sphinx/reference/spec5-packages.md`

## Contract

PoPS is not a Python numerical solver. Python authors typed objects, validates
their contracts, generates or selects compiled C++ routes, and binds them to the
C++/Kokkos/MPI runtime.

The public flow is:

```python
case = pops.Case(layout=layout)
compiled = pops.compile(case, backend=Production())
sim = pops.bind(compiled, state=initial_state, params=params)
sim.run(t_end=1.0, cfl=0.5)
```

There must not be Python callbacks inside cell loops, face loops, Riemann
solves, Krylov iterations, AMR reflux, halo exchange, or field solves.

## Names versus behavior

Strings are stable user names:

```python
case.block("electrons", physics=model)
T = Program("ssprk3")
```

Behavior is selected by typed objects:

```python
from pops.codegen import Production
from pops.mesh import CartesianMesh
from pops.mesh.layouts import Uniform, AMR
from pops.numerics.riemann import HLL
from pops.numerics.reconstruction import MUSCL
from pops.numerics.reconstruction.limiters import Minmod
from pops.solvers.elliptic import GeometricMG

layout = Uniform(CartesianMesh(n=256, L=1.0, periodic=True))
spatial = pops.FiniteVolume(
    riemann=HLL(),
    reconstruction=MUSCL(limiter=Minmod()),
)
field_solver = GeometricMG(tolerance=1.0e-10)
backend = Production()
```

Do not document public APIs that choose behavior with string tokens such as the
old geometric-MG solver token, AMR target token, or production backend token.
Those strings may still exist internally as native IDs, but descriptors own the
public choice.

## Case assembly

`pops.Case` is the top-level authoring object:

```python
case = (
    pops.Case(layout=layout, name="diocotron")
    .block("electrons", physics=model, spatial=spatial)
    .field(poisson)
    .time(program)
    .output(output_policy)
)
```

The case is inert. It describes layout, blocks, fields, parameters, outputs, and
time. `pops.compile` lowers it. `pops.bind` attaches runtime data and returns the
runnable simulation facade.

## Mesh layout

Mesh structure is not a compile target.

```python
from pops.mesh import CartesianMesh
from pops.mesh.layouts import Uniform, AMR
from pops.mesh.amr import Refine, RegridEvery

mesh = CartesianMesh(n=128, L=1.0, periodic=True)

uniform = Uniform(mesh)
amr = AMR(
    base=mesh,
    max_levels=2,
    ratio=2,
    regrid=RegridEvery(4),
    refine=Refine.on("density").above(0.05),
)
```

The public API passes `layout=uniform` or `layout=amr`. Internal C++ routes such
as `System` and `AmrSystem` are selected by lowering.

## Time programs

`pops.time` is the language for building time programs. Ready-made schemes live
in `pops.lib.time`.

```python
from pops.time import Program

T = Program("ssprk3")
U = T.state("U", block="electrons")

T.define(U.stage(1), U.n + T.dt * R(U.n))
T.define(U.stage(2), 0.75 * U.n + 0.25 * (U.stage(1) + T.dt * R(U.stage(1))))
T.define(U.next, (1.0 / 3.0) * U.n + (2.0 / 3.0) * (U.stage(2) + T.dt * R(U.stage(2))))
T.commit("electrons", U.next)
```

Use `pops.lib.time.ssprk3(...)`, `pops.lib.time.strang(...)`, or the other
library macros when the scheme is already provided.

## Inspection

Compiled problems and runtime simulations must be inspectable:

```python
print(compiled)
compiled.inspect()
compiled.arguments()
compiled.dump_ir()
compiled.dump_cpp()

memory = compiled.estimate_memory(grid=mesh, layout=layout)
print(memory)

sim = pops.bind(compiled, state=state)
with sim.profile(Profile.Advanced()) as prof:
    sim.run(t_end=1.0, cfl=0.5)
prof.summary().print()
```

Inspection is part of the user API because generated C++ and GPU memory use
must be debug-visible.
