# Add a new run recipe

Named application recipes should live outside the reusable core when they are
scenario-specific. In this repository, a recipe should demonstrate the public
compiled problem route.

```python
module = build_model().to_module()
program = build_program(module)

compiled = pops.compile_problem(model=module, program=program, backend=Production(), layout=layout)
sim = pops.System(n=mesh.n, L=mesh.L, periodic=mesh.periodic)
sim.install(
    compiled,
    instances={"ne": {"model": module, "initial": ne0, "spatial": spatial}},
    solvers={"phi": field_solver},
)
sim.step_cfl(0.4)
```

Keep scenario names, parameter defaults, and validation data in the scenario
repository. Keep generic descriptors and C++ runtime support in PoPS.
