# M18/W09: finite-quadrature entropy closure, first public realization

`DiscreteEntropyQuadrature(nodes, weights, basis)` validates positive weights,
finite coefficients and full row rank, then builds ordinary symbolic expressions
for

\[
 p_i = w_i\exp\!\left(\sum_a \lambda_a m_{ai}\right), \qquad
 R_a(\lambda;u) = \sum_i m_{ai}p_i-u_a.
\]

The public example `examples/migration/scientific/api040_m18_entropy.py` fixes
five nodes `(-1,-.5,0,.5,1)`, weights `(.1,.2,.4,.2,.1)` and basis
`(1,v,v²)`. It specifies 20 moderate, strictly interior target cells. Two
state blocks give `LocalResidual` one three-component dual unknown and an exact
read-only target capture. The target is initialized on its own State block and
is never projected from the solve or committed again. `LocalNewton` has 12 iterations, backtracking and an
original-residual tolerance of `2e-11`. A failed solve is consumed through
`FailRun`; the step must not publish the dual block or advance the clock, while
the read-only target remains unchanged.

The new public `pops.math.exp` is a distinct `Exp` expression node. Its
structural/CSE key and closed Program DAG opcode are `exp`, with derivative
`exp(a)*a'`. The existing Program serialization schema 3 and native package
ABI 6 remain unchanged: opcode identity is self-describing inside the
versioned Program payload, and the generated numerical source participates
in artifact identity. An older emitter rejects the unknown opcode rather than
interpreting it as `pow` or a polynomial. An installed source/SDK identity
must still be rebuilt and authenticated before any runtime qualification.
`where` retains lazy branch evaluation; an active overflow is a nonfinite
original residual, never clamped to zero or a finite ceiling.

The independent pure oracle enumerates nonnegative basic feasible populations
and solves a two-dimensional nullspace max-min LP. It labels targets
`interior`, `boundary`, or `outside` independently of Newton convergence.
For interior targets it also runs a damped dual Newton, checks every original
moment residual and positive population, and compares primal entropy along
nonzero moment-nullspace perturbations. The `outside` witness has mass one,
first moment zero and second moment 1.1, impossible because all nodes satisfy
`v²<=1`. The `boundary` witness is the `v=1` ray: feasible with a zero-population
representation but no finite dual multiplier attaining it. Only this
initial-data oracle assigns those labels. The native local solver reports its
generic numerical status; it does not yet provide a dynamic per-cell cone
certificate or separate native `E-CLOSURE-INFEASIBLE` and `BOUNDARY` codes.

Source tests and an isolated host C++ probe of the emitted public residual
are separate from installed PoPS evidence. The installed integration witness
is collected but has not been run on a rebuilt package in this branch. It is
designed to check all 20 accepted cells and a one-cell outside-cone refusal,
including all-rank failure collection, the committed dual and read-only target
buffers, time and temporal
envelope across two failed attempts. It does not claim near-boundary solver
convergence, AMR, GPU, arbitrary quadrature rank or a transport PDE.
