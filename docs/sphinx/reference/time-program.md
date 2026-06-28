# Time programs

`pops.time` is the temporal language. It builds a typed IR for one time step.
The generated step runs C++-side.

Ready-made schemes live in `pops.lib.time`.

## Minimal program

```python
from pops.time import Program
from pops.numerics.terms import Flux

T = Program("forward_euler")
U = T.state("U", block="plasma")

fields = T.solve_fields(U.n)
R = T.rhs(state=U.n, fields=fields, terms=[Flux()])

T.define(U.next, U.n + T.dt * R)
T.commit("plasma", U.next)
```

`T.define` materializes a state-valued IR node. `T.commit` marks the state that
replaces the live block at the end of the step.

## Version handles

`T.state("U", block="plasma")` returns a temporal handle family:

| Handle | Meaning |
| --- | --- |
| `U.n` | Current state at the start of the step. Read-only. |
| `U.stage(k)` | Named intermediate state. Must be defined once with `T.define`. |
| `U.next` | End-of-step state. Must be defined before `T.commit`. |
| `U.prev` / `U.prev(k)` | History state for multistep schemes. Read-only. |

Handles contain no arrays. They resolve to IR values.

## Ready schemes

Use `pops.lib.time` for provided schemes:

```python
from pops.time import Program
from pops.lib.time import ssprk3

T = Program("advance")
ssprk3(T, "plasma")
```

Available families:

| Function | Family |
| --- | --- |
| `forward_euler` | Explicit Euler |
| `ssprk2`, `ssprk3` | Strong-stability-preserving RK |
| `rk4`, `rk`, `explicit_rk` | Classic and tableau-driven explicit RK |
| `adams_bashforth`, `adams_bashforth2` | Multistep explicit schemes |
| `bdf` | BDF program builder |
| `strang`, `lie` | Split flows |
| `imex_local`, `imex_local_linear` | Local implicit source treatment |
| `condensed_schur` | Schur-condensed source-stage builder |
| `predictor_corrector_local_linear` | Predictor-corrector local-linear macro |

The functions add IR nodes to a `Program`. They do not compute arrays.

## RHS terms

The public RHS builder takes typed terms:

```python
from pops.numerics.terms import Flux, SourceTerm

electric = SourceTerm("electric")
R = T.rhs(state=U.n, fields=fields, terms=[Flux(), electric])
```

The string in `SourceTerm("electric")` names a source declared by the model. The
behavior is the typed `SourceTerm` object.

## Operator handles

Programs can call model operators by handle:

```python
fields_op = model.field_operator(...)
rate_op = model.rate(...)

T = Program("operator_step")
U = T.state("U", block="plasma")

fields = T.call(fields_op, U.n)
R = T.call(rate_op, U.n, fields)
T.define(U.next, U.n + T.dt * R)
T.commit("plasma", U.next)
```

Do not document string operator references. Strings may create operators; handles
reference them.

## Histories

Multistep schemes store and read history through handles or the lower-level
history API:

```python
T.keep_history("plasma.U", copy="current")
older = U.prev
```

History storage is runtime-owned C++ state.

## Solves

Linear and nonlinear solves are described in the program and executed by compiled
solver routes:

```python
from pops.solvers.krylov import GMRES

phi = T.solve_linear(problem, method=GMRES(tolerance=1.0e-10))
```

Users configure provided solvers. Python never runs a Krylov loop.

## Inspection

Useful methods:

```python
print(T)
T.inspect()
T.dump_ir()
T.estimate_report()
T.scratch_liveness()
T.buffer_reuse_report()
```

Generated-source inspection happens on the compiled handle:

```python
compiled.dump_ir()
compiled.dump_cpp("generated")
compiled.estimate_memory(grid=mesh)
```
