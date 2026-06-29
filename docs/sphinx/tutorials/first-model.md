# First model

Author the model as typed physics, lower it to a module, build a time program,
compile the pair, then install the compiled artifact.

```python
from pops.physics import Model
from pops.math import ddt, div
from pops.time import Program
from pops.lib.time import ssprk3

m = Model("first_model")
U = m.state("U", components=["rho"], roles={"rho": "density"})
(rho,) = U
flux = m.flux("F", on=U, x=[rho], y=[0.0 * rho], waves={"x": [1.0 + 0.0 * rho]})
m.rate("explicit_rate", ddt(U) == -div(flux))

module = m.to_module()
program = Program("advance").bind_operators(module)
ssprk3(program, "plasma", rhs_operator=module.operator_registry().get("explicit_rate"))

compiled = pops.compile_problem(model=module, program=program, backend=Production(), layout=layout)
sim = pops.System(n=mesh.n, L=mesh.L, periodic=mesh.periodic)
sim.install(compiled, instances={"plasma": {"model": module, "initial": rho0, "spatial": spatial}})
sim.step_cfl(0.4)
```
