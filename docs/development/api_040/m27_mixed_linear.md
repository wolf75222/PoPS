# M27: periodic linear mixed field witness

The API 0.4.0 corpus fixes the stable linear subcase
`mu = c - epsilon² Delta(c)`, `epsilon = 0.08`, periodic boundaries,
`N = 16` and `dt = 0.01`, followed by `N = 32, 64`. The native witness in
`examples/migration/scientific/api040_m27_mixed_linear.py` keeps **both**
unknowns and both backward Euler relations:

```text
c_next - c_old = dt Delta_h(mu_next)
mu_next - c_next + epsilon² Delta_h(c_next) = 0
```

It advances ten steps to `t = 0.1` on the real one-dimensional periodic
Uniform layout. This end time and the smooth two-mode initial profile are
witness choices; the corpus does not fix them. The initial values are exact
cell-volume means of `0.4 + 0.1 cos(2 pi x) + 0.05 sin(4 pi x)`.

`CellCenteredGeneralCoupled` selects the same native cell-centred,
matrix-free field stencil with an **explicit finite-general coefficient
admission**. The ordered principal matrix is nonsymmetric and indefinite:
`D = [[0, dt], [-epsilon², 0]]` for `(c, mu)`; the reaction matrix is
`R = [[1, 0], [-1, 1]]`. The default `CellCenteredSecondOrder` retains its
strict SPD admission. `GMRES` is required by the general linear-operator
properties. No global dense matrix or fourth-order substitute is used in
production. The native prepared Krylov path verifies the original physical
residual before publishing a solve result.

`FieldSolution.cell_mean_state(field[c], target=current.next)` is a narrow
version-1 projection. It authenticates the source field observation, the
registered cell-centred method, constant coefficients, the exact State load,
block, StateSpace, physical support, cell-volume-mean representation and
evaluation point. It then uses the existing one-component scalar-field commit
route. The chemical potential remains a separate native solved field and is
saved from accepted history slot 1; no host-side reconstruction of `mu` is
passed off as a native observation. Different sampling, support transfers,
state-dependent coefficients and nonlinear projections require separate
rules and are refused here.

The independent NumPy oracle solves the **two original equations** for each
Fourier mode using a 2-by-2 matrix. It uses the periodic second-difference
symbol `-4 N² sin²(pi k/N)` and an unrelated direct cell-mean integral for
initial-data verification. The saved native `c` and `mu` are reopened and
checked at every accepted step. Both original discrete residuals are computed
from these saved fields. Mass and the quadratic discrete energy
`(mean(c²) + epsilon² mean((c[i+1]-c[i])²)/h²)/2` are checked. This energy
belongs to the linear stable subcase; it does not certify the nonlinear
double-well model or phase separation.

Criteria fixed before native execution: initial cell means `2e-14`; every
saved `c` and `mu` against the Fourier oracle `3e-10`; each original relation
`1e-10`; mass drift `2e-12`; per-step energy increase `3e-12`; clock
`3e-14`. The final energy must be strictly lower. The same equations are
also authored with `(mu,c)` unknown order. A separate one-iteration GMRES
case requires a collective `iteration_limit` refusal with State, history,
diagnostics, step and time unchanged.

Source-only checks cover authoring, validate, resolve and complete Program
emission for the listed grids and both unknown orders. These checks do **not**
authenticate an installed native artifact. The installed native reception is
`tests/python/integration/runtime/test_api040_m27_mixed_linear_runtime.py`,
run by the repository's `docs/development/api_040/run_installed_checks.py`
after the root rebuilds the package for Dim1.

On 2026-09-30, the installed Dim1 CPU/Kokkos/MPICH reception passed both
tests: the four complete ten-step trajectories and the one-iteration
collective refusal. The earlier `restart=30, max_iter=400` configuration
reached `iteration_limit` at step 5 for N=64; its failed log and the completed
N=16/32 saved arrays are retained in workspace
`outputs/installed-m26-m27-native-dim1-fc0-20260930`.
The example now exposes `solver_restart` and defaults to 128 retained Krylov
vectors, with the same 400-iteration budget and `rel_tol=abs_tol=1e-12`.
Neither original equation nor any saved-state acceptance threshold changed.
This is an explicit solver configuration for the witness, not a production
DOF limit or a dense PDE matrix. The successful receipt is
`outputs/installed-m27-native-dim1-restart128-fc0-20260930`. The separate
MPI2 group `outputs/installed-m26-m27-mpi2-fc0-20260930` passed three tests
on both ranks, with installation authentication before and after, unchanged
test sources and identical rank test inventories. This group also receives
the finite M26 witness; it does not qualify its aggregation-diffusion PDE.
