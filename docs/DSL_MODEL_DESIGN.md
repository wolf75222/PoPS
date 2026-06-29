# DSL model design

`pops.physics.Model` is a writing facade. It creates a `pops.model.Module`.
Compilation and runtime execution are separate responsibilities.

```python
from pops.physics import Model
from pops.math import ddt, div

m = Model("example")
U = m.state("U", components=["rho"], roles={"rho": "density"})
(rho,) = U
F = m.flux("F", on=U, x=[rho], y=[0.0 * rho], waves={"x": [1.0 + 0.0 * rho]})
m.rate("explicit_rate", ddt(U) == -div(F))

module = m.to_module()
```

A time program consumes operator handles from the module:

```python
from pops.time import Program

ops = module.operator_registry()
program = Program("advance").bind_operators(module)
U = program.state("U", block="plasma")
R = program.call(ops.get("explicit_rate"), U.n)
program.define(U.next, U.n + program.dt * R)
program.commit("plasma", U.next)
```

The compiled artifact combines the model and the program:

```python
compiled = pops.compile_problem(model=module, program=program, backend=Production(), layout=layout)
sim = pops.System(n=mesh.n, L=mesh.L, periodic=mesh.periodic)
sim.install(compiled, instances={"plasma": {"model": module, "initial": U0}})
sim.step_cfl(0.4)
```

`pops.physics` does not compile, import the native extension, allocate runtime
data, or execute kernels. It only authors model/operator IR.
