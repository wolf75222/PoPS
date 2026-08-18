# PO-07 — elliptic tolerance sweep (1-d PO-01 oracle)

Phase 1 code-verification case. In-memory manufactured 1-d trigonometric
Poisson plus an algebraic-vs-discretization error model. Public Case
authoring is present. No native compile, no bind, no `pops.run`, no ROMEO.

| Field | Content |
|---|---|
| Identifier | `PO-07` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | \(-\phi''=\rho\) on a 1-d periodic interval. Manufactured \(\phi=\sin(2\pi x)\), \(\rho=(2\pi)^2\phi\), \(E=-d\phi/dx=-2\pi\cos(2\pi x)\). Combined error \(\max(C h^2,\mathrm{tol})\). |
| Oracle | Reuses the PO-01 1-d \(\phi\). `phi_exact` / `rhs_exact` / `e_exact` plus `discretization_error(n)`, `algebraic_error(tol)=tol`, `combined_error=max(disc,alg)` in `exact.py`. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Samples are uniform cell centers. |
| Parameters | Dimensionless. Resolutions \(N=16,32,64,128\). Relative tolerances \(10^{-6},10^{-8},10^{-10},10^{-12}\). Scale \(C=0.04\) for the manufactured \(E\propto h^2\) floor. |
| Native dimensions | `POPS_NATIVE_DIM=1` only. This worktree does not load a native artifact. |
| Required capabilities | Periodic Cartesian Poisson, FFT or GeometricMG, constant-kernel gauge, relative elliptic tolerance. MPI off. |
| Configurations | Uniform 1-d cells. Public Case: `-laplacian(phi)==rhs`, `FFT()`, `Periodic`, `ConstantNullspace`, `MeanValueGauge(0)`. Sweep is in-memory `combined_error`. |
| Diagnostics | Combined error vs tolerance at fixed \(N\); discretization plateau vs \(N\); Task 15 observed order of the tight-tol floor \(E\propto h^2\). |
| Thresholds | Spatial order threshold \(1.8\) for a declared second-order scheme when algebraic error is below the floor. Plateau at \(N\) equals \(C/N^2\). |
| Proves | Algebraic error equals the requested tolerance; combined error plateaus on the discretization floor; the floor drops when \(N\) doubles; schema-valid campaign report without a solver. |
| Does not prove | Native elliptic solve, iteration counts, 2-d/3-d Poisson, AMR, coupling, MPI, performance. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. Suites: nightly, weekly, release. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
