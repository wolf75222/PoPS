# Time schemes

Time integration is expressed as a compiled `pops.time.Program`.

`pops.time` contains the language. `pops.lib.time` contains ready schemes.

```python
from pops.time import Program
from pops.lib.time import ssprk3

time = Program("advance")
ssprk3(time, "plasma")
case = case.time(time)
```

Manual schemes use temporal handles:

```python
from pops.numerics.terms import Flux

T = Program("forward_euler")
U = T.state("U", block="plasma")
fields = T.solve_fields(U.n)
R = T.rhs(state=U.n, fields=fields, terms=[Flux()])
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

The CFL value is a runtime policy passed to `sim.run`.

```python
sim.run(t_final=1.0, cfl=0.4)
```

The time program should not reject a run only because the CFL is not a compile
time default.
