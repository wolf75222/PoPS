# Define initial conditions

Initial state arrays are passed through `sim.install(..., instances=...)`.

For a scalar block, the array is usually `(n, n)`. For a conservative vector
state, the array is usually `(ncomp, n, n)`.

```python
sim.install(
    compiled,
    instances={
        "ne": {
            "model": module,
            "initial": ne,
            "spatial": spatial,
        },
    },
)
```

For several blocks, pass one entry per block:

```python
sim.install(
    compiled,
    instances={
        "electrons": {"model": electron_module, "initial": Ue0, "spatial": electron_spatial},
        "ions": {"model": ion_module, "initial": Ui0, "spatial": ion_spatial},
    },
)
```

After installation, advance with explicit CFL steps:

```python
t_final = 0.1
cfl = 0.4

while sim.time() < t_final:
    sim.step_cfl(cfl)
```
