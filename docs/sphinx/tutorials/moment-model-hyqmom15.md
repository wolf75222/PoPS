# HyQMOM15 moment model

Moment helpers build typed moment bases, closures, realizability projectors, and
wave-speed contracts. A ready HyQMOM model must still lower to a
`pops.model.Module`.

```python
from pops.lib.models.moments import HyQMOM15

module = HyQMOM15.safe_default().to_module()
program = Program("advance").bind_operators(module)
ssprk3(program, "moments", rhs_operator=module.operator_registry().get("explicit_rate"))

compiled = pops.compile_problem(model=module, program=program, backend=Production(), layout=layout)

sim = pops.System(n=mesh.n, L=mesh.L, periodic=mesh.periodic)
sim.install(
    compiled,
    instances={"moments": {"model": module, "initial": M0, "spatial": spatial}},
    params={"q_over_m": -1.0},
)
sim.step_cfl(0.4)
```

For moment systems, positivity is not enough. The module/descriptors must carry
realizability, closure, wave-speed, projection, and moment-ordering contracts so
validation can reject incompatible numerical routes before codegen.
