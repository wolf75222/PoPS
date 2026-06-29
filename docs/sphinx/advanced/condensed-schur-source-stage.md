# Condensed Schur source stage

Condensed Schur stages are ready-made time macros. They belong in
`pops.lib.time`, not in the core `pops.time` language.

```python
from pops.lib.time import condensed_schur
from pops.time import Program
from pops.solvers.krylov import GMRES

program = Program("condensed_schur")
condensed_schur(
    program,
    block="plasma",
    alpha=1.0,
    method=GMRES(),
    tol=1.0e-10,
    max_iter=200,
)
```

The macro expands to `pops.time.Program` nodes. Solver and preconditioner choices
are typed descriptors and lower to compiled C++ routes.

Use inspection to check the expansion:

```python
compiled = pops.compile_problem(model=module, time=program, backend=Production(), layout=layout)
compiled.dump_ir()
```
