# Bounded independent AMR Newton/GMRES source and math review

Frozen source: `9fb7e8fb64c3e3de289d9c139ba310e9e47e43a6`.
Private branch: `codex/api040-sol61-amr-gmres-math-review` in
`work/PoPS-sol61-ale-canonical-serial-offline`. All earlier ALE receipts and
branches are preserved. Tests/docs only; no PoPS/native import, build, JIT,
installed environment, donor or MAIN edits.

Root's actual wave6e native receipt has the serial/MPI failure
`OriginalAmrNonlinearResidualTwoResolutionsAndPermutation` with reason
`amr_field_newton_gmres_breakdown`. It also has a distinct pending-Outcome
fixture abort. These native failures were not rerun here and remain failures.

## Source path and limits of the reason

`prepared_amr_field_residual.hpp` supplies `-F` as the linear right-hand side,
and computes the central difference of the original `+F` for its JVP. The
Newton workspace adds the correction. Those signs are coherent after the
earlier correction; this review does not reset them.

The AMR inner product in `amr_field_newton_krylov.hpp` loops all components and
levels, uses the active-cell mask and level measure, and counts a replicated
level on one physical owner before its collective reduction. The original
operator's inner product uses the same masks/measures/ownership. No missing
component or covered-parent counting defect was demonstrated here.
The scalar composite operator restricts covered values and fills halos before
application. The original local equation and the final original-equation
residual are still checked on the actual candidate.

`solve_linear_` uses one MGS pass. It has no DGKS reorthogonalization like the
separate `generic_krylov.hpp` implementation. Its `LinearResult` false flag can
mean nonfinite beta, invalid Givens magnitude, invalid triangular pivot/value,
or exhaustion of the 240-iteration budget. All produce the same public
`gmres_breakdown` reason. That reason alone cannot establish a true Arnoldi
breakdown or identify the native failure cause. Euler's separate frozen
failure-only diagnostic `75c1d2c5f59fcda88a394d6caecccfea5fe64380` is therefore
needed for the next native reception; this review does not receive that
unexecuted diagnostic as a numerical repair.

## Concrete projected-convergence counterexample

There is a bounded internal acceptance defect independent of the native
failure diagnosis. Immediately after `update_correction_`, the AMR GMRES
returns `converged=true` when its rotated/projected RHS estimate is below stop.
It skips the existing actual `rhs - JVP(correction)` computation in that
branch. In finite precision, the projected estimate need not bound the actual
linear residual.

The independent test uses only two-dimensional standard Arnoldi/MGS/Givens
algebra and explicit binary64 accumulation. It does not construct field states
or implement an AMR/runtime substitute. Its matrix is exactly symmetric SPD
when interpreted as the stored binary64 values:

```
A = [[ 64000000000000.36, -47999999999999.52],
     [-47999999999999.52,  36000000000000.64]]
b = [0.3, 0.7]
```

The exact determinant is `1633280000000000051/16384 > 0`; the leading minor is
positive and all data and resulting coefficients are finite. The authored
linear relative tolerance remains `1e-5`:

| Quantity | Result |
|---|---:|
| Stop = `1e-5 * norm(b)` | 7.6157731058639085e-6 |
| Projected residual estimate after two columns | 0 |
| Binary64 `norm(b - A*x)` | 0.0032211762700137757 |
| Residual recomputed with exact rational products/sums of the stored floats | 0.004824114810754357 |

The computed correction is `[0.443235655046038, 0.5909808733947204]`.
The exact recomputation avoids claiming that cancellation in a second floating
matrix multiplication makes this a valid solve. Its residual exceeds stop by
more than 633 times. A true residual check using the unchanged stop would reject
that linear-convergence claim. The probe pins and inspects the **historical
source branch**, rather than requiring a future modified header to remain
identical or receiving a new repair from an old counterexample.

This is a very ill-conditioned matrix, approximately condition 1e14. It is
**not a reproduction or causal attribution** of the actual moderate
three-component AMR witness at 16/32 cells. The nonlinear Armijo and final
original-equation guards already prevent this internal false convergence from
being sufficient to publish an inaccurate original-equation solution. They
must remain in place.

## Bounded recommendation and reception still needed

Recheck the actual JVP correction residual against the original linear stop
before returning linear convergence, including apparent lucky breakdown.
Do not raise tolerances, discard a component, reset norms, substitute an easier
physical equation, or claim that reorthogonalization alone proves the native
problem solved. Measure the failing native Newton iteration, completed Krylov
columns, stop, projected residual, beta and failing pivot/nonfinite category
before choosing a correction. Central differencing
`(.5/h)*Fplus - (.5/h)*Fminus` also has cancellation sensitivity, but this
review has not demonstrated it as the native failure cause.

The separate Outcome-abort lifecycle remains Euler's investigation and must
be received independently of a numerical correction. No production change is
made in this review.

## Executed checks

```sh
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m pytest -q -p no:cacheprovider tests/review/test_sol61_amr_gmres_projection_math.py
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m ruff check tests/review/test_sol61_amr_gmres_projection_math.py
rtk git diff --check
```

Result: **2 source/math PASS**, Ruff PASS, diff-check PASS. No native, MPI,
Kokkos, GPU or AMR execution qualification is asserted.
