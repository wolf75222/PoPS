# Multi-block AMR

A multi-block AMR case has several physics blocks on one shared hierarchy.

```python
case = (
    pops.Case(layout=layout, name="two_species")
    .block("ions", physics=ions, spatial=ion_spatial)
    .block("electrons", physics=electrons, spatial=electron_spatial)
    .field(poisson)
    .time(time)
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

A field RHS such as charge density is assembled from the named contributing
blocks:

```python
from pops.fields.rhs import ChargeDensity

rhs = ChargeDensity.from_blocks("ions", "electrons")
```

The field problem is attached once to the case. The runtime assembles the
coupled RHS from the block states.

## Time programs

A time program may reference multiple blocks through their handles:

```python
from pops.time import Program

T = Program("coupled_step")
ions = T.state("U_i", block="ions")
electrons = T.state("U_e", block="electrons")
```

When the time program needs one coupled field solve from simultaneous stage
states, use the multi-block field solve operator or handle supplied by the model
layer. The field solve still lowers to C++.

## AMR tags

Tags from all blocks are combined into one hierarchy. The hierarchy is then used
for every block, so reflux and averaging remain conservative per block.
