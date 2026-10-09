# Remaining original AMR Newton failure: diagnostic freeze

Base: `9fb7e8fb64c3e3de289d9c139ba310e9e47e43a6`.

The authentic wave6e run fails the nonconstant original nonlinear case in both
serial and MPI2. The identity original equation with arbitrary component widths
passes. This distinguishes the remaining failure from the previously repaired
Newton sign error and from a failure specific to MPI ownership.

The workspace previously returned the same `amr_field_newton_gmres_breakdown`
reason for exhausted iterations, zero Arnoldi columns, zero triangular pivots,
and nonfinite evaluations. This freeze records the exact exit category, Newton
iteration, completed Arnoldi columns, JVP evaluations, requested linear stop,
last explicitly recomputed residual norm, projected residual norm and last pivot.
Floating-point witnesses use hexadecimal notation. The failure status and action
remain Breakdown/RejectAttempt. No algebra operation, norm, budget, tolerance,
finite-difference step, authored equation, or publication rule changes.

The central JVP currently scales the two original evaluations separately before
subtracting them. Cancellation at small residuals, direction-dependent nonlinear
finite differences and one-pass modified Gram--Schmidt are hypotheses; none is
established as the cause by this source-only freeze. The native diagnostics must
identify the actual exit before a numerical correction is justified.

The publication fixture has a distinct contract gap: a pre-Accept authority
validation failure intentionally leaves an otherwise solved Outcome pending for
retry, but an irreversibly revoked lease cannot retry and the current API refuses
RejectAttempt/FailRun for solved reports. Leaving that Outcome unconsumed invokes
the deliberate destructor abort. A separate explicit collective candidate
discard operation is required; it must preserve the numerical report and never
call Accept. This freeze does not change SolveOutcome.

Validation: source comparison covers every numerical statement in the actual
GMRES loop and triangular update; full actual C++ test translation-unit syntax
uses the real GoogleTest, Kokkos and MPI headers. Native execution remains root
owned. Run the narrow source tests with:

```
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q tests/review/test_sol61_amr_gmres_diagnostics.py
```
