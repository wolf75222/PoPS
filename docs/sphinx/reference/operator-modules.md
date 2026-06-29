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
module = physics_model.lower()
ops = module.operator_registry()
explicit_rate = ops.get("explicit_rate")
```

The exact handle-producing helper depends on the authoring facade. The rule is
stable: create names once, then pass handles.

## Program calls

```python
from pops.time import Program

T = Program("step").bind_operators(module)
U = T.state("U", block="plasma")

fields_op = ops.get("fields_from_state")
fields = T.call(fields_op, U.n)
rate = T.call(explicit_rate, U.n, fields)
T.define(U.next, U.n + T.dt * rate)
T.commit("plasma", U.next)
```

Do not call operators by free string names in public examples. The internal IR
may lower handles to native IDs, but the public API passes handles.

## Rate operators

Prefer a declared rate operator handle when a program builds a residual:

```python
rate = model.rate_operator("electric_rate", flux=True, sources=["electric"])
R = T.call(rate, U.n, fields)
```

Primitive RHS builders are internal to `pops.lib.time` and tests.

## Inspection

```python
compiled = pops.compile_problem(model=module, time=T, layout=layout)
compiled.inspect()
compiled.dump_ir()
compiled.dump_cpp()
```

Inspection should show which operator handle lowered to which C++ route.
