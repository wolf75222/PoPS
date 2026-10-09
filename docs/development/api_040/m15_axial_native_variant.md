# M15: fixed-calendar native axial B.1 variant

This is a **genuine spatial Dim=1**, one-velocity, order-four, five-moment
experiment. It does not qualify the M15 Fox–Laurent multi-order recurrence.
The physical closure remains Appendix B.1:
`S50 = S30 (5 S40 - 3 S30² - 1)/2`. Its implementation, density/variance/Hankel
admission and signed speeds from the full flux Jacobian are unchanged.
No clipping, floor, projection or Gaussian replacement is permitted.

Before native reception, fix the periodic interval `[0,1]`, `N=32,64,128`,
FirstOrder reconstruction, HLL and Forward Euler, `dt=1/(100 N)` and `t_end=.02`
(64/128/256 accepted steps; no rejection). This is an explicitly selected
FixedDt variant. The existing `build_case(cells)` default retains AdaptiveCFL(.1).
Both canonical storage and the permutation `(4,1,3,0,2)` are executed.

The initial distribution is

```text
f(x,v) = (.6 + .08 cos(2πx)) Normal(v; mean=-.4, variance=.3)
       + (.4 + .06 sin(2πx)) Normal(v; mean= .7, variance=.2).
```

Both weights and variances are strictly positive. Each raw moment is integrated
over the spatial cell analytically: the trigonometric factors carry `sinc(1/N)`.
The native initial global state is compared with these means before evolution.
Center samples are explicitly distinguished by a pure quadrature counter-test.

The separate NumPy oracle imports no PoPS. It expresses the same B.1 fifth
central moment rationally, differentiates with complex steps, computes the
non-symmetric Jacobian eigenvalues with NumPy, then applies periodic HLL/FE.
References for every grid and their domain/Courant checks are computed before
the first native compile. This is an independent oracle for the **discrete
method**, not an exact solution of the nonlinear PDE or an order-of-convergence
claim. Initial/final Hankel positivity is checked; the oracle checks every step.
Native face recovery checks its active states before use; final native cells
are checked again after their NPZ archive is reopened.

The fixed bounds, declared in `api040_m15_axial_oracle.py`, are:

| Criterion | Bound |
|---|---:|
| Initial cell averages, max norm | 2e-14 |
| Final state versus independent HLL/FE, max norm | 3e-11 |
| Each of five periodic moment integrals | 2e-12 |
| Native final time | 2e-14 |
| Canonical/permuted state agreement | 3e-12 |
| Imaginary part of oracle spectrum | 2e-11 |
| Oracle `dt * max_face_speed / h` preflight budget | .1 |

The preflight maximum is about .01609 for the specified data, below .1; even
the conservative sum of two incident face maxima is below .03218. This is
preflight evidence, not a theorem of nonlinear positivity for arbitrary B.1
states. Density, `H2=M0 M2-M1²` and `H3=det([M_(i+j)]_(i,j=0..2))` retain the
literal tests `rho>0`, `H2>0`, `H3>=0` with no relaxed domain tolerance.

Three finite invalid initial vectors isolate the guards: `(-1,0,-1,0,0)` violates
only density; `(1,0,0,0,1)` only positive variance; `(1,0,1,0,.5)` only H3.
The same artifact first runs valid data, then each negative must report native
recovery rejection or the exact finite-volume admission status, on every rank,
with state and time unchanged if bind returned a runtime. A compiler error or
unrelated bind error cannot qualify. The current native recovery status groups
these predicates together; this test does not invent distinct per-predicate
status codes.

All ranks participate in global gathers. Only rank zero writes/opens the NPZ
and JSON, and its assertions/writing failures are broadcast. Receipts retain
state hashes, run reports, execution contexts and exact native bytes. The
scientific runner authenticates before and after, explicitly selecting Dim=1
for `m15-axial-b1`; all existing case defaults remain Dim=2.

After root's dedicated Dim=1 build/install (no Dim=2 substitution):

```sh
env -u PYTHONPATH PYTHONNOUSERSITE=1 POPS_NATIVE_DIM=1 \
  POPS_REQUIRE_NATIVE_TESTS=1 CONDA_PREFIX="$POPS_PREFIX" \
  Kokkos_ROOT="$POPS_PREFIX" POPS_KOKKOS_ROOT="$POPS_PREFIX" \
  POPS_INCLUDE="$POPS_PREFIX/lib/python3.12/site-packages/pops/include" \
  OMP_NUM_THREADS=1 "$POPS_PREFIX/bin/python" \
  docs/development/api_040/run_installed_checks.py --output /tmp/m15-dim1-runtime \
  --test tests/python/integration/runtime/test_m15_axial_b1_runtime.py

env -u PYTHONPATH CONDA_PREFIX="$POPS_PREFIX" Kokkos_ROOT="$POPS_PREFIX" \
  POPS_KOKKOS_ROOT="$POPS_PREFIX" \
  POPS_INCLUDE="$POPS_PREFIX/lib/python3.12/site-packages/pops/include" \
  "$POPS_PREFIX/bin/python" docs/development/api_040/run_scientific_checks.py \
  --case m15-axial-b1 --ranks 1 --threads 1 --output /tmp/m15-dim1-full
```

Repeat the scientific command with `--ranks 2` and a fresh output directory for
the separate MPI receipt. The runtime test is intentionally Dim=1-only; it is
not a passing Dim=2 compatibility test. No installed-native result has been
produced by the preparation or source checks in this change.
