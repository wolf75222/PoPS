# Hybrid native plus DSL model

A hybrid model combines ready C++ bricks and DSL-authored operators into one
`pops.model.Module`. Once the module exists, the runtime route is unchanged.

```python
module = hybrid_model.to_module()
program = Program("advance").bind_operators(module)

compiled = pops.compile_problem(model=module, program=program, backend=Production(), layout=layout)
sim = pops.System(n=mesh.n, L=mesh.L, periodic=mesh.periodic)
sim.install(compiled, instances={"plasma": {"model": module, "initial": U0, "spatial": spatial}})
sim.step_cfl(0.4)
```

The hybrid boundary is inside model construction, not in runtime assembly.
