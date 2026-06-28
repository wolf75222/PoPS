# Bound simulation

The public way to obtain a runnable simulation is `pops.bind`.

```python
compiled = pops.compile(case, backend=Production())
sim = pops.bind(compiled, state={"plasma": U0})
```

`bind` uses the case layout to select the runtime route:

- `Uniform(mesh)` builds a single-level runtime;
- `AMR(mesh, ...)` builds an adaptive runtime.

The returned object exposes high-level runtime actions:

```python
sim.run(t_end=1.0, cfl=0.4)
sim.write("out", format="npz")
sim.time()
sim.mass("plasma")
sim.density("plasma")
```

The implementation has lower-level runtime methods because pybind and tests need
them. User documentation should not use them to assemble a case.

## Inputs

`pops.bind` accepts:

- `state`: initial block arrays;
- `params`: runtime parameter overrides;
- `aux`: named static aux fields;
- `solvers`: optional field-solver overrides;
- `cadence`: optional compiled-time cadence.

The compiled handle declares required inputs through `compiled.arguments()`.
`bind` validates missing inputs before mutating the runtime.

## Inspection

```python
print(sim)
compiled.arguments()
compiled.inspect()
```

These reports should be small and array-free.
