# Configure outputs and diagnostics

Install output policies with the compiled problem, then use the runtime facade
for readback and file output.

```python
sim.install(compiled, instances={"ne": {"model": module, "initial": ne0}},
            outputs=outputs)
```

## Read diagnostics

```python
rho = sim.density("ne")
phi = sim.potential()
mass = sim.mass("ne")
t = sim.time()
```

These readbacks are diagnostics and visualization aids. They are not Python
callbacks in the numerical loop.

## Write fields to disk

```python
sim.write("out/state", format="vtk", step=42)
sim.write("out/state", format="npz", step=42)
sim.checkpoint("out/restart.npz")
```

Output/checkpoint policies that run during stepping are installed as typed
policies through `sim.install(..., outputs=...)`.
