# Define initial conditions

Bind the initial state of a block from numpy arrays, then advance the run. This
page assumes you already have a compiled `Case`; see
[Configure a simulation](../simulation/index.md) for that. The layout convention is row-major
`(ny, nx)`: index a field as `arr[j, i]`, where `j` is the row (`y` axis) and `i` is the column
(`x` axis).

## Set a scalar density

Pass a block state through `pops.bind`. For a scalar block, `ARR` is an `(n, n)` array.

1. Build the cell-center coordinates and the field.

   ```python
   coord = (np.arange(n) + 0.5) / n * L
   ```

2. Mesh the coordinates with `indexing="xy"` so both arrays have shape `(ny, nx)`.

   ```python
   xx, yy = np.meshgrid(coord, coord, indexing="xy")
   ```

3. Fill the density array and bind it.

   ```python
   ne = np.full((n, n), 1e-3)
   ```

   ```python
   sim = pops.bind(compiled, state={"ne": ne})
   ```

For a periodic Poisson, fix a neutralizing background equal to the mean (`n_i0 = ne.mean()`)
so the right-hand side is solvable.

## Set a fluid state from primitives

For fluid models, bind the conservative state expected by the compiled block. Helper functions may
prepare that array from primitive variables (`rho`, `u`, `v`, `p`) before binding.

```python
U0 = conservative_from_primitives(rho=rho0, u=u0, v=v0, p=p0)
sim = pops.bind(compiled, state={"electrons": U0})
```

The conservative state uses the block's component order and an `(ncomp, n, n)` layout.

## Advance and check the result

After binding, `sim.run(t_end=final_time, cfl=CFL)` advances the simulation by CFL-limited steps up to the requested final time.

```python
sim.run(t_end=0.1, cfl=0.4)
```

Read `sim.density(NAME)` for the field, `sim.potential()` for `phi`, and `sim.mass(NAME)` for
the total mass. The mass is the conservation invariant; with periodic advective transport its
drift stays near machine roundoff.

## Next steps

- [Configure a simulation](../simulation/index.md) for the spatial scheme, time policy and Poisson.
- [A->Z tutorial](../getting-started/tutorial.md) for an end-to-end diocotron run.
