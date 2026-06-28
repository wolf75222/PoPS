# Add a new model

A model defines local physics: state variables, fluxes, sources, field coupling,
wave speeds, roles, capabilities, and diagnostics. It does not define the mesh,
the AMR layout, the time scheme, or the runtime.

Use this flow:

1. Author the physics.
2. Add the model to a `pops.Case`.
3. Compile the case.
4. Bind data and run the C++ runtime.

## Author the physics

Use `pops.physics.Model` when the model is not already available as a library
preset:

```python
from pops.physics import Model
from pops.math import div, ddt

m = Model("scalar_transport")
U = m.state("U", components=["rho"], roles={"rho": "density"})
rho = U[0]

F = m.flux("F", on=U, x=[rho], y=[0.0 * rho], waves={"x": [1.0], "y": [0.0]})
m.rate("explicit_rate", ddt(U) == -div(F))

model = m.lower()
```

`pops.physics` writes `pops.model` IR. It does not compile or run.

## Choose numerical descriptors

```python
from pops.numerics.riemann import Rusanov
from pops.numerics.reconstruction import MUSCL
from pops.numerics.reconstruction.limiters import Minmod

spatial = pops.FiniteVolume(
    riemann=Rusanov(),
    reconstruction=MUSCL(limiter=Minmod()),
)
```

The numerical choices are typed objects. Do not select behavior with strings.

## Assemble a case

```python
from pops.mesh import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.time import Program
from pops.lib.time import ssprk3

mesh = CartesianMesh(n=128, L=1.0, periodic=True)
program = Program("ssprk3")
ssprk3(program, "plasma")

case = (
    pops.Case(layout=Uniform(mesh), name="scalar_case")
    .block("plasma", physics=model, spatial=spatial)
    .time(program)
)
```

## Compile and run

```python
from pops.codegen import Production

compiled = pops.compile(case, backend=Production())
print(compiled)
print(compiled.arguments())

sim = pops.bind(compiled, state={"plasma": U0})
sim.run(t_end=0.1, cfl=0.4)
```

The bind step owns runtime data. The generated or selected C++ route owns the
cell loops.

## Use AMR

AMR is a layout change, not a different model API:

```python
from pops.mesh.layouts import AMR
from pops.mesh.amr import Refine, RegridEvery

layout = AMR(
    base=mesh,
    max_levels=2,
    ratio=2,
    regrid=RegridEvery(4),
    refine=Refine.on("density").above(0.05),
)
```

Use the same `Case`, descriptors, and time program with `layout=layout`.
