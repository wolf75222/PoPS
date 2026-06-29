# System install

The public runtime flow installs one compiled problem artifact on an explicit
runtime facade:

```python
compiled = pops.compile_problem(model=module, time=program, backend=Production(), layout=layout)

sim = pops.System(n=mesh.n, L=mesh.L, periodic=mesh.periodic)
sim.install(
    compiled,
    instances={
        "plasma": {
            "model": module,
            "initial": U0,
            "spatial": spatial.FiniteVolume(reconstruction=Minmod(), riemann=Rusanov()),
        },
    },
    params=params,
    aux=aux,
    solvers=solvers,
)
sim.step_cfl(0.4)
```

For an adaptive layout, use the same compiled artifact contract and install it
on `pops.AmrSystem(...)`. Users do not pass target strings; the layout
descriptor determines the generated artifact ABI.

## Inputs

`sim.install` accepts:

- `instances`: initial block arrays plus per-instance model and spatial descriptors;
- `params`: runtime parameter overrides;
- `aux`: named static auxiliary fields;
- `solvers`: field solver descriptors;
- `outputs`: optional output and checkpoint policies.

The compiled handle declares required inputs through `compiled.arguments()`.
`sim.install` validates missing inputs before mutating the runtime.

## Execution

`sim.step_cfl(cfl)` advances one CFL-limited macro step in C++/Kokkos/MPI.
When a script needs a final time, keep the final time explicit:

```python
while sim.time() < t_final:
    sim.step_cfl(cfl)
```

Avoid positional wrappers that hide what a number means. Prefer named variables
such as `t_final` and `cfl`.

## Inspection

```python
print(compiled)
print(sim)
compiled.arguments()
compiled.inspect()
```

These reports are small and array-free.
