# C16–C19: a small Newton step is not residual convergence

## Concrete accepted counterexample

The public request is `LocalResidual(R, seed, captures={"fixed": old})`, with
`R(z;d)=z*z-d`, `seed=2`, fixed datum `d=2`, and
`LocalNewton(tolerance=1e-12, relative_tolerance=0, step_tolerance=1)`.
One Newton update gives approximately `z=1.5`, `R=.25`, and step norm `.5`.
The original provider returned `kConverged` solely because `.5 <= 1`, despite
the freshly evaluated original equation failing its declared tolerance.

The independent test compiled the actual public residual and controls emitted
by PoPS, then executed `solve_prepared_local_nonlinear` from the real native
header. It reproduced the false success: **1 failed in 1.53 s**, receipt
`outputs/t3-original-residual-red.xml`. The final fixture additionally performs
public Case validation and resolution before emission.

## Shared correction

The existing residual test remains the only success criterion. If that test
fails and the optional step threshold is reached, the provider now returns
`kSafeguardFailure`: this is an unsuccessful stagnation stop. The failed result
retains its candidate and diagnostics for inspection, while its solve report
does not authorize publication. No status enum, residual tolerance, solver
algorithm, physical model, publication callback, or ProgramContext was changed.
The LocalNewton public docstring explains this behavior.

The change applies to the shared prepared local nonlinear provider, including
the generic LocalResidual route and other existing consumers of that provider.
The success check precedes the stagnation check, so a satisfied original
equation is still accepted even when its step is small or zero.

## Validation and remaining scope

The seven independent public tests cover the scalar counterexample, small fixed
damping, a coupled two-component unknown with both original equations checked,
frozen captures independent of a changed seed, normal convergence, a root
already satisfying its residual, and an invalid capture. The full pure-source
selection plus earlier domain/capture tests passes: **132 passed in 6.37 s**,
receipt `outputs/t3-original-residual-source-green.xml`.

Two durable gtests were added to `test_newton_robustness.cpp`. Their exact test
bodies were also compiled and run in a small host TU against the actual provider
and report header: **2/2 in binary64 and 2/2 in binary32**. This is not execution
of the entire runtime test target. The failed-result test explicitly verifies
`report.valid()` and `!report.solved_value_available()`.

An initial wider command also included the legacy process-isolated script
`test_time_local_newton.py`. It failed before computation because the spawned
process imported the installed runtime without selecting a native dimension:
**1 failed, 132 passed** (`outputs/t3-original-residual-suite-green.xml`, whose
filename does not override its recorded failure). It was not retried against an
old installed native header. No whole-runtime, MPI, GPU or heavy build was run.

This closes one actual false-acceptance seam, not the whole heterogeneous solve
profile. The public coupled-vector test is one two-component State; arbitrary
products of distinct global unknowns and an original full-system residual after
general condensation are not established here. Condensation currently does not
carry an arbitrary original coupled residual descriptor, so inventing one from
a particular physical model would not be a generic repair. The native relative
criterion's pre-existing reference normalization also remains unchanged.
