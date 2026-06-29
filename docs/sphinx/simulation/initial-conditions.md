# Initial conditions

Initial conditions are runtime inputs to `sim.install`.

```python
coord = (np.arange(n) + 0.5) / n * L
xx, yy = np.meshgrid(coord, coord, indexing="xy")
r = np.hypot(xx - 0.5 * L, yy - 0.5 * L)
ne = np.full((n, n), 1e-3)
ne[(r > 0.15) & (r < 0.20)] = 1.0

sim.install(compiled, instances={"ne": {"model": module, "initial": ne, "spatial": spatial}})
```

For vector conservative states, pass `(ncomp, n, n)`:

```python
sim.install(
    compiled,
    instances={
        "electrons": {
            "model": electron_module,
            "initial": U0,
            "spatial": electron_spatial,
        },
    },
)
```

The layout convention is row-major `(ny, nx)`: the first index is the slow `y`
axis and the second is the fast `x` axis.
