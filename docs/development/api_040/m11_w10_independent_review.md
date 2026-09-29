# Independent review of the finite M11/W10 witness

Reviewed `4e0d363` against the closed W10 corpus data. The source friction
matrix is the graph Laplacian of the specified symmetric off-diagonal
coefficients `(2,1,3)`: it has rank two and kernel `(1,1,1)`. The full 4×4
augmented system is invertible. The example keeps the target force as a
capture distinct from the zero solve seed. It checks the **original**
`L*j-force` and `sum(j)` before both commits, so a solved augmented system with
nonzero multiplier does not authorize an incompatible force. The NumPy oracle
independently constructs L from the off-diagonal matrix and solves the full
system. Its one-cell incompatible force has an augmented solution but an
original residual of order `0.1`, far above the declared `1e-12` guard.

The native fixture had a collective-control flaw: local clock and temporal
rollback assertions preceded `_root_check`, a broadcast. A single-rank
assertion failure could strand peers in that broadcast. This review changes
only the fixture so those assertions are gathered from every rank before the
next collective check. It preserves the mathematical data, failure diagnostic,
bitwise state rollback, thresholds and same-artifact fresh-bind check.

The seven pure/source tests pass, and both native variants collect. Native
serial/MPI execution, numerical status, and rollback on an installed artifact
remain for central reception. This is a finite algebraic W10 witness, not a
qualification of the full M11 Poisson–Maxwell–Stefan PDE.
