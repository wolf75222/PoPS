# Add a native brick

Native bricks are C++ implementations selected by typed Python descriptors.

1. Add the C++ implementation and native routing ID.
2. Add a Python descriptor with requirements, capabilities, options, and
   validation.
3. Lower the descriptor through `compile_problem` / `sim.install`.
4. Add tests that exercise the descriptor through a compiled problem artifact.

```python
compiled = pops.compile_problem(model=module, time=program, backend=Production(), layout=layout)
sim = pops.System(n=mesh.n, L=mesh.L, periodic=mesh.periodic)
sim.install(compiled, instances={"plasma": {"model": module, "initial": state, "spatial": spatial}})
sim.step_cfl(0.4)
```

The descriptor may lower to a native token internally, but public Python should
not expose string selectors.
