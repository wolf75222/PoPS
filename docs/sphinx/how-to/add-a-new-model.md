# Add a new model

1. Author the physics with `pops.physics.Model` or build a `pops.model.Module`
   directly.
2. Build a `pops.time.Program` that calls typed operator handles.
3. Compile the model/program pair with `pops.compile_problem`.
4. Install the compiled artifact on `pops.System` or `pops.AmrSystem`.

```python
module = physics_model.to_module()
program = Program("advance").bind_operators(module)

compiled = pops.compile_problem(model=module, time=program, backend=Production(), layout=layout)
sim = pops.System(n=mesh.n, L=mesh.L, periodic=mesh.periodic)
sim.install(compiled, instances={"plasma": {"model": module, "initial": U0, "spatial": spatial}})
sim.step_cfl(0.4)
```

The model is responsible for physical operators and capabilities. The spatial
descriptor is responsible for numerical reconstruction/Riemann choices.
