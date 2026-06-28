# Simulation

A simulation is created by binding a compiled case.

```python
compiled = pops.compile(case, backend=Production())
sim = pops.bind(compiled, state=state, params=params)
sim.run(t_end=1.0, cfl=0.4)
```

The case owns the description. The bound simulation owns runtime data and
execution.

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
