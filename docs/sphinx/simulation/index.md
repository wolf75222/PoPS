# Simulation

A simulation starts from a compiled problem artifact:

```python
compiled = pops.compile_problem(model=module, time=program, backend=Production(), layout=layout)
sim = pops.System(n=mesh.n, L=mesh.L, periodic=mesh.periodic)
sim.install(compiled, instances={"plasma": {"model": module, "initial": U0}})
sim.step_cfl(0.4)
```

The model/program pair owns the description. The runtime facade owns arrays,
field solvers, output policies, profiling, and execution.

```{toctree}
:maxdepth: 1

system
spatial-schemes
time-schemes
substeps-stride-multirate
initial-conditions
outputs-diagnostics
```

## Runtime rule

Python may pass arrays, parameters, descriptors, and policies. It must not run
numerical loops. Transport, sources, field solves, reductions, halos, AMR, and
output are C++/Kokkos/MPI work.
