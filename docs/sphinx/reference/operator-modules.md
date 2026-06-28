# Operator modules

`pops.model` is the operator-first IR. It is the explicit layer that
`pops.physics` lowers to and that `pops.time.Program` consumes.

## What an operator module contains

An operator module declares:

- state spaces;
- field spaces;
- operator signatures;
- operator handles;
- requirements and capabilities.

Operator names are strings at declaration time. Program references use handles.

```python
from pops.model import OperatorHandle

module = physics_model.lower()
explicit_rate = OperatorHandle("explicit_rate")
```

The exact handle-producing helper depends on the authoring facade. The rule is
stable: create names once, then pass handles.

## Program calls

```python
from pops.time import Program

T = Program("step").bind_operators(module)
U = T.state("U", block="plasma")

fields = T.solve_fields(state=U.n)
rate = T.call(explicit_rate, U.n, fields)
T.define(U.next, U.n + T.dt * rate)
T.commit("plasma", U.next)
```

Do not call operators by free string names in public examples. The internal IR
may lower handles to native IDs, but the public API passes handles.

## RHS terms

Use `P.rhs(..., terms=[...])` when the program builds a residual directly:

```python
from pops.numerics.terms import Flux, SourceTerm

R = T.rhs(
    state=U.n,
    fields=fields,
    terms=[Flux(), SourceTerm("electric")],
)
```

The old boolean style is not public documentation.

## Inspection

```python
compiled = pops.compile(case)
compiled.inspect()
compiled.dump_ir()
compiled.dump_cpp()
```

Inspection should show which operator handle lowered to which C++ route.
