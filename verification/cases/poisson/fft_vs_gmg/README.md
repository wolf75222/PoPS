# PO-05 — FFT vs GMG cross-oracle (1-d PO-01 reuse)

Phase 0 code-verification case. Reuses the PO-01 trigonometric Poisson
oracle. In-memory spectral FFT solve versus the discrete -Δ residual that
a GeometricMG stub would drive to zero. Public Case authoring is present.
No native compile, no bind, no `pops.run`, no ROMEO.

| Field | Content |
|---|---|
| Identifier | `PO-05` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | \(-\phi''=\rho\) on a 1-d periodic interval. Manufactured \(\phi=\sin(2\pi x)\), \(\rho=(2\pi)^2\phi\), \(E=-d\phi/dx=-2\pi\cos(2\pi x)\). Spectral solve: \(k^2\hat\phi=\hat\rho\) with \(k=0\to 0\). Discrete residual: \((-\Delta_h\phi)-\rho\). |
| Oracle | PO-01 `phi_exact` / `rhs_exact` / `e_exact` loaded with `load_sibling_module` from `verification/cases/poisson/periodic_trig/exact.py`. Local `spectral_solve(rhs)` and `gmg_stub_residual(phi, rhs)`. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Samples are uniform cell centers. Mean-zero gauge. |
| Parameters | Dimensionless. Cross-oracle on \(N=32\). Spectral recovery tolerance \(10^{-12}\). |
| Native dimensions | `POPS_NATIVE_DIM=1` only. This worktree does not load a native artifact. |
| Required capabilities | Periodic Cartesian Poisson, FFT or GeometricMG, constant-kernel gauge. MPI off. |
| Configurations | Uniform 1-d cells. Public Cases: `-laplacian(phi)==rhs`, `FFT()` or `GeometricMG()`, `Periodic`, `ConstantNullspace`, `MeanValueGauge(0)`. Comparison is in-memory. |
| Diagnostics | Task 2 L∞ of mean-free spectral \(\phi\) vs analytic \(\phi\). L2 of the discrete \(-\Delta\) residual of analytic \(\phi\) vs manufactured \(\rho\) (FD truncation; not a gate). |
| Thresholds | Spectral \(\phi\) recovers analytic \(\phi\) (mean-zero) to \(10^{-12}\) on \(N=32\). Discrete residual is reported and is not an acceptance gate. |
| Proves | PO-01 reuse via `load_sibling_module`. Spectral FFT recovers mean-zero analytic \(\phi\). Discrete residual of analytic \(\phi\) is the known FD truncation. Schema-valid campaign report without a solver. |
| Does not prove | Native FFT or GeometricMG solve, discrete-symbol FFT vs MG identity, 2-d/3-d, AMR, coupling, MPI, performance. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
