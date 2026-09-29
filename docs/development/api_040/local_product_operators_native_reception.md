# T3 v2 independent installed reception: local source/apply products

This is a test-only reception tranche, based on integrated source `cfef849`, which
contains the T3 v2 local-operator implementation and the sparse Program detachment
fix. No production code, native header, environment or numerical threshold changes.
The helper's optional `return_parameters=True` returns exact `block[param]` handles;
existing calls still return the original three values.

## Manufactured original system

The public helper uses three distinct Models and Blocks with widths 1, 2 and 4,
homonymous parameters named `gain`, old-value captures, a nonconstant candidate
argument `P.source(2*z)` and `P.apply(z)`. For component c numbered from zero,

`R_ic(x) = 4*g_i*x_ic² + (c+1)*x_ic + 0.1*sum_jd(x_jd) - old_ic`.

The independent NumPy oracle imports no PoPS code and never calls a generated
residual. On a 4×4 grid it manufactures the positive solution
`x_ic = .25 + .07*i + .035*c + .04*x_cell + .06*y_cell`, with cell-centre
coordinates. It constructs `old` from these equations and checks the native result
against both the manufactured solution and **every original equation**, not a
reduced system or a small Newton step. The source helper has fixed seed 1, so the
captured old value is independent from the algorithmic seed.

Two non-default parameter vectors, `(.2,.65,1.1)` and `(1.3,.35,.8)`, bind the same
compiled artifact through three distinct qualified handles. They produce distinct
captures. Both canonical and reversed Block insertion / unknown-key order are
compiled. This is a permutation of block/key authority, not a claim of arbitrary
component permutation or mesh redistribution.

Thresholds fixed before native reception: maximum absolute original residual
strictly below `1e-11`, manufactured solution absolute error at most `1e-10`, no
relative error allowance, all candidate values finite. The positive branch has
Jacobian `diag(8*g_i*x_ic+c+1) + 0.1*ones`, positive definite there; the test does
not assume universal invertibility of local products or uniqueness of all roots
outside that branch. The candidate-dependent source factor four distinguishes
evaluation at `2*z` from a frozen or incorrectly unscaled source.

## Algebraically incompatible equation and atomic refusal

At cell `[1,2]` set all seven captured old values to -100 while keeping gains
`(.2,.65,1.1)`. For N=7, summing all original equations gives seven independent
quadratics with linear coefficients `b_ic = c+1+.7`. Completing their squares,

`sum(R) >= -sum_ic(b_ic²/(16*g_i)) - sum_ic(old_ic)`.

Here `-sum(old)=700`, the quadratic cost is approximately `4.493334790209815`, and
the lower bound is approximately `695.5066652097902`, strictly positive for
**every real candidate**. Thus no simultaneous zero exists; the expected refusal
is not based on an arbitrary guess about Newton iteration count. A pure oracle
test evaluates the exact minimizer and confirms this sum identity.

The native witness gathers all exception types before asserting, requires
`RuntimeError` with the genuine `coupled_implicit failed:` outcome diagnostic on
every rank, and collectively reads all three block states. It requires exact
state restoration, time zero, macro-step zero and restoration of the accepted
temporal envelope. Uniform CPU commits share one clock; independent asynchronous
block clocks are not claimed. The refusal is repeated on the unchanged runtime.
Finally a **new public bind of the same artifact** with consistent data succeeds.
This is artifact reuse after failure, not a successful retry or parameter mutation
of the incompatible runtime.

## Evidence

Executed source/pure checks: 17 passed, covering the five new oracle/public-helper
tests and the twelve existing local-product-operator tests. Both public variants
pass validate → resolve → ProgramModelGraph → System emission of the seven-wide
native local nonlinear provider. Ruff passes. The two integration variants are
collected successfully; no native compilation, JIT, MPI or GPU execution was
performed by this worker.

Central reception target (serial and MPI2, with the normal runner timeout):
`tests/python/integration/runtime/test_local_product_operators_runtime.py`.
Preserve its all-rank diagnostics and lower-bound properties in the JUnit receipt.
It uses the repository's compile-once MPI helper, authenticated ExecutionContext
and collective root assertion helper. Qualify only the exact installed artifact
that runs it; source emission alone is not native acceptance.
