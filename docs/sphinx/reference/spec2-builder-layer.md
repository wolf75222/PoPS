# Builder layer

The builder layer is the explicit IR API below `pops.physics`. It is useful for
library authors, generated macros, and tests that need to inspect exact program
nodes.

## Operators are handles

Declare operators on a model/module, then pass the returned handle into the
program:

```python
from pops.model import OperatorHandle

rate_op = OperatorHandle("explicit_rate")

T = Program("step").bind_operators(module)
U = T.state("U", block="plasma")
fields = T.solve_fields(state=U.n)
R = T.call(rate_op, U.n, fields)
```

Strings name operators at declaration time. Handles reference operators later.

## Define temporal versions

```python
T.define(U.stage(1), U.n + T.dt * R)
T.define(U.next, U.stage(1))
T.commit("plasma", U.next)
```

`T.define` lowers to the existing SSA program values. It does not allocate Python
runtime data.

## Direct RHS composition

```python
from pops.numerics.terms import Flux, SourceTerm

R = T.rhs(
    state=U.n,
    fields=fields,
    terms=[Flux(), SourceTerm("electric")],
)
```

Use this only when no predeclared operator handle is more appropriate.
