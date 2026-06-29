# Write a model with bricks

Ready bricks are typed descriptors or ready model assemblies. They still enter
the same compiled problem route.

```python
module = ready_model.to_module()

program = Program("advance").bind_operators(module)
ssprk3(program, "electrons", rhs_operator=module.operator_registry().get("explicit_rate"))

compiled = pops.compile_problem(model=module, time=program, backend=Production(), layout=layout)

sim = pops.System(n=mesh.n, L=mesh.L, periodic=mesh.periodic)
sim.install(
    compiled,
    instances={"electrons": {"model": module, "initial": ne0, "spatial": spatial}},
    solvers={"phi": field_solver},
)
sim.step_cfl(0.4)
```

The model brick owns physics. The spatial descriptor owns reconstruction and
Riemann choices. The runtime only installs and executes the compiled artifact.
