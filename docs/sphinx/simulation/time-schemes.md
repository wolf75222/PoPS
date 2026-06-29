# Time schemes

Time integration is expressed as a compiled `pops.time.Program`.

`pops.time` contains the language. `pops.lib.time` contains ready schemes.

```python
from pops.time import Program
from pops.lib.time import ssprk3

time = Program("advance")
ssprk3(time, "plasma")
compiled = pops.compile_problem(model=module, time=time, backend=Production(), layout=layout)
```

Manual schemes use temporal handles:

```python
T = Program("forward_euler")
U = T.state("U", block="plasma")
T.bind_operators(model)

ops = model.operator_registry()
fields_from_state = ops.get("fields_from_state")
rate = ops.get("explicit_rate")

fields = T.call(fields_from_state, U.n)
R = T.call(rate, U.n, fields)
T.define(U.next, U.n + T.dt * R)
T.commit("plasma", U.next)
```

## Provided scheme families

| Function | Package |
| --- | --- |
| `forward_euler` | `pops.lib.time` |
| `ssprk2`, `ssprk3` | `pops.lib.time` |
| `rk4`, `rk`, `explicit_rk` | `pops.lib.time` |
| `adams_bashforth`, `bdf` | `pops.lib.time` |
| `strang`, `lie` | `pops.lib.time` |
| `imex_local`, `imex_local_linear` | `pops.lib.time` |
| `condensed_schur` | `pops.lib.time` |

These functions build IR nodes. They are not alternate runtime steppers.

## CFL

The CFL value is passed to each runtime step.

```python
sim.step_cfl(0.4)
```

The time program should not reject a run only because the CFL is not a compile
time default.
