# Custom solvers

A solver in adc is a typed brick. It can be native C++, written in the Python DSL and
generated to C++, provided by an external C++ library, or specialized into `problem.so`. In
every case it runs in C++ and uses the shared HPC primitives (dot, norm, axpy, MPI
reductions, the scratch manager, the profiler); Python never iterates a Krylov loop.

## Native solvers (today)

The matrix-free Krylov solvers are native C++ free functions in
`include/adc/numerics/elliptic/linear/generic_krylov.hpp`:

| Solver | Symbol |
| --- | --- |
| Richardson | `adc::richardson_solve` |
| CG | `adc::cg_solve` |
| BiCGStab | `adc::bicgstab_solve` |
| GMRES | `adc::gmres_solve` |

They are named by descriptors ({doc}`typed-bricks`): `adc.lib.solvers.CG()`,
`adc.lib.solvers.BiCGStab()`, `adc.lib.solvers.GMRES()`, `adc.lib.solvers.Richardson()`. A
compiled time Program drives them through `P.solve_linear(...)` ({doc}`time-program`); the
elliptic field solve uses the geometric multigrid (`adc::GeometricMG`).

## Generated solvers (design)

A solver can be written in the Python DSL and generated to C++ -- it builds an IR, it does not
compute in Python:

```python
@adc.lib.solver(name="richardson", signature=(A, b, x0) >> x)
def richardson(ctx, A, b, x0, omega, tol, max_iter):
    x = x0
    r = b - A(x)
    res = ctx.norm2(r)
    it = ctx.scalar_int(0)
    with ctx.while_(ctx.logical_and(res > tol, it < max_iter)):
        x = x + omega * r
        r = b - A(x)
        res = ctx.norm2(r)
        it = it + 1
    return x
```

The lowering emits C++ that uses the core primitives (dot / norm / axpy / linear_combine /
MPI reductions / scratch / matrix-free apply / profiler). Modes: `native`, `generated`,
`library`, `specialized`, `auto`.

```{admonition} Status
:class: note
The native Krylov solvers and the `adc.lib.solvers` descriptors exist. The solver DSL
(`@adc.lib.solver`), `compile_library`, the specialization modes and external C++ solver
registration are follow-ups; the Program `solve_linear` over the native Krylov solvers is the
supported path today.
```
