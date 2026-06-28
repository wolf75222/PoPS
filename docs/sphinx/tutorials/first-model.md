# Run your first model

This tutorial runs a small hyperbolic-elliptic model through the public PoPS
flow:

```text
Case -> compile -> bind -> sim.run
```

## Build and check the module

```bash
bash scripts/setup_env.sh
conda activate pops
pip install .
python -c "import pops; pops.doctor()"
```

## Author the model

```python
from pops.physics import Model
from pops.math import div, ddt

m = Model("scalar_transport")
U = m.state("U", components=["rho"], roles={"rho": "density"})
rho = U[0]

F = m.flux("F", on=U, x=[rho], y=[0.0 * rho], waves={"x": [1.0], "y": [0.0]})
explicit_rate = m.rate("explicit_rate", ddt(U) == -div(F))
model = m.lower()
```

## Assemble the case

```python
import numpy as np
import pops
from pops.mesh import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.numerics.riemann import Rusanov
from pops.numerics.reconstruction import MUSCL
from pops.numerics.reconstruction.limiters import Minmod
from pops.time import Program
from pops.lib.time import ssprk2
from pops.codegen import Production

mesh = CartesianMesh(n=64, L=1.0, periodic=True)
spatial = pops.FiniteVolume(
    riemann=Rusanov(),
    reconstruction=MUSCL(limiter=Minmod()),
)

program = Program("ssprk2")
ssprk2(program, "plasma")

case = (
    pops.Case(layout=Uniform(mesh), name="first_model")
    .block("plasma", physics=model, spatial=spatial)
    .time(program)
)
```

## Compile, bind, and run

```python
compiled = pops.compile(case, backend=Production())
print(compiled)

rho0 = np.ones((64, 64), dtype=float)
sim = pops.bind(compiled, state={"plasma": rho0})
sim.run(t_end=0.1, cfl=0.4)
```

Use `compiled.inspect()`, `compiled.arguments()`, and
`compiled.estimate_memory(...)` before running larger cases.
