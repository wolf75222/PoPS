# Multi-species and multi-block programs

A multi-species run is a compiled model/program artifact installed with several
named block instances. Each block supplies its own initial state, model module,
and spatial descriptor. Shared fields, sources, and diagnostics reference
blocks by stable names.

```python
compiled = pops.compile_problem(model=shared_module, program=program,
                                backend=Production(), layout=layout)

sim = pops.System(n=mesh.n, L=mesh.L, periodic=mesh.periodic)
sim.install(
    compiled,
    instances={
        "electrons": {"model": electron_module, "initial": Ue0, "spatial": electron_spatial},
        "ions": {"model": ion_module, "initial": Ui0, "spatial": ion_spatial},
    },
    solvers={"phi": poisson_solver},
)
```

Field right-hand sides can assemble contributions from several blocks:

```python
from pops.fields.rhs import ChargeDensity

rhs = ChargeDensity.from_blocks("electrons", "ions")
```

## Program handles

```python
from pops.time import Program

T = Program("two_species_step").bind_operators(shared_module)
ops = shared_module.operator_registry()

e = T.state("Ue", block="electrons")
i = T.state("Ui", block="ions")

fields = T.call(ops.get("fields_from_blocks"), e.n, i.n)
Re = T.call(ops.get("electron_rate"), e.n, fields)
Ri = T.call(ops.get("ion_rate"), i.n, fields)

T.define(e.next, e.n + T.dt * Re)
T.define(i.next, i.n + T.dt * Ri)
T.commit("electrons", e.next)
T.commit("ions", i.next)
```

The state versions are handles, not runtime arrays. `sim.install` provides the
actual state data.

## AMR

Multi-block runs use the same `Uniform(...)` or `AMR(...)` layout descriptor as
single-block runs. AMR policies should tag fields or roles with enough block
context to avoid ambiguity.
