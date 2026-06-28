# Multi-species and multi-block programs

A multi-species model is a `pops.Case` with several named blocks. Each block has
its own physics model and spatial descriptors. Shared fields, sources, and
diagnostics reference blocks by stable names.

## Assemble blocks

```python
case = (
    pops.Case(layout=layout, name="two_species")
    .block("electrons", physics=electron_model, spatial=electron_spatial)
    .block("ions", physics=ion_model, spatial=ion_spatial)
    .field(poisson)
    .time(program)
)
```

`poisson` can assemble its right-hand side from multiple blocks:

```python
from pops.fields.rhs import ChargeDensity

rhs = ChargeDensity.from_blocks("electrons", "ions")
```

## Program handles

```python
from pops.time import Program

T = Program("two_species_step")
e = T.state("Ue", block="electrons")
i = T.state("Ui", block="ions")

fields = T.solve_fields_from_blocks([e.n, i.n])
Re = electron_rate(e.n, fields)
Ri = ion_rate(i.n, fields)

T.define(e.next, e.n + T.dt * Re)
T.define(i.next, i.n + T.dt * Ri)
T.commit_many({"electrons": e.next, "ions": i.next})
```

The state versions are handles, not runtime arrays. Bind provides the actual
state data.

## AMR

Multi-block cases use the same `Uniform(...)` or `AMR(...)` layout descriptor as
single-block cases. AMR policies should tag fields or roles with enough block
context to avoid ambiguity.
