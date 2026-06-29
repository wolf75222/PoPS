# Write a model with the physics DSL

The physics DSL is a writing facade for `pops.model.Module`.

```python
from pops.physics import Model
from pops.math import ddt, div

m = Model("dsl_model")
U = m.state("U", components=["rho"], roles={"rho": "density"})
(rho,) = U
F = m.flux("F", on=U, x=[rho], y=[0.0 * rho], waves={"x": [1.0 + 0.0 * rho]})
m.rate("explicit_rate", ddt(U) == -div(F))

module = m.to_module()
```

Build and run through the compiled problem route:

```python
program = Program("advance").bind_operators(module)
ssprk3(program, "ne", rhs_operator=module.operator_registry().get("explicit_rate"))

compiled = pops.compile_problem(model=module, time=program, backend=Production(), layout=layout)
sim = pops.System(n=mesh.n, L=mesh.L, periodic=mesh.periodic)
sim.install(compiled, instances={"ne": {"model": module, "initial": ne0, "spatial": spatial}})
sim.step_cfl(0.4)
```
