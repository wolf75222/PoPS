# Multi-block AMR

A multi-block AMR run installs one compiled problem artifact on a shared AMR
hierarchy:

```python
compiled = pops.compile_problem(model=shared_module, time=program,
                                backend=Production(), layout=layout)
sim = pops.AmrSystem(n=mesh.n, L=mesh.L)
sim.install(
    compiled,
    instances={
        "ions": {"model": ions_module, "initial": Ui0, "spatial": ion_spatial},
        "electrons": {"model": electrons_module, "initial": Ue0, "spatial": electron_spatial},
    },
    solvers={"phi": poisson_solver},
)
```

The blocks share:

- mesh hierarchy;
- patch layout;
- field solves;
- halo/regrid orchestration;
- output/checkpoint cadence.

Each block keeps its own:

- physics model;
- spatial descriptor;
- state array;
- source and projection contracts;
- field contribution.

## Field coupling

A field RHS such as charge density is assembled from named contributing blocks:

```python
from pops.fields.rhs import ChargeDensity

rhs = ChargeDensity.from_blocks("ions", "electrons")
```

The field solve lowers to C++; Python only describes the coupling.

## Time programs

```python
from pops.time import Program

T = Program("coupled_step")
ions = T.state("U_i", block="ions")
electrons = T.state("U_e", block="electrons")
```

When the time program needs one coupled field solve from simultaneous stage
states, use a typed operator handle supplied by the model layer.

## AMR tags

Tags from all blocks are combined into one hierarchy. The hierarchy is then used
for every block, so reflux and averaging remain conservative per block.
