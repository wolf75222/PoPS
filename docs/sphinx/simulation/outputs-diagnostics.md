# Outputs and diagnostics

The runtime exposes array readback and diagnostics after a compiled problem has
been installed.

```python
m0 = sim.mass("ne")
sim.step_cfl(0.4)

rho = sim.density("ne")
phi = sim.potential()
print("mass drift:", abs(sim.mass("ne") - m0))
```

## Time loops

Keep final time and CFL explicit:

```python
t_final = 0.2
cfl = 0.4

while sim.time() < t_final:
    sim.step_cfl(cfl)
```

## Write to disk

```python
sim.write("out/state", format="vtk", step=42)
sim.checkpoint("out/run.ckpt")
```

`format="vtk"` writes visualization output. `format="npz"` writes a compressed
NumPy archive. `format="hdf5"` is available when the optional HDF5 stack is
enabled.

`pops.System(layout=AMR(...))` exposes the same high-level output contract with AMR-aware
layout and patch data.

Condensed API reference: [api](../reference/python-api.md). Open a dump in
ParaView: [visualize with ParaView](../how-to/visualize-with-paraview.md).
