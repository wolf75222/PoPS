# PO-01 — periodic trigonometric Poisson (1-d reduction)

Phase 1 code-verification case. In-memory manufactured 1-d trigonometric
Poisson only. Public Case authoring is present. No native compile, no bind,
no `pops.run`, no ROMEO.

| Field | Content |
|---|---|
| Identifier | `PO-01` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | \(-\phi''=\rho\) on a 1-d periodic interval. Manufactured \(\phi=\sin(2\pi x)\), \(\rho=(2\pi)^2\phi\), \(E=-d\phi/dx=-2\pi\cos(2\pi x)\). Reduction of \(\phi=\sin(2\pi x)\sin(4\pi y)\cos(2\pi z)\). |
| Oracle | `phi_exact` / `rhs_exact` / `e_exact` in `exact.py`. Optional 2-d restriction `phi_exact_2d(x,y)=sin(2πx)sin(4πy)`. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Samples are uniform cell centers. |
| Parameters | Dimensionless. Manufactured order series \(N=16,32,64,128\). Optional \(C h^2\) perturbation of \(\phi\). |
| Native dimensions | `POPS_NATIVE_DIM=1` only. This worktree does not load a native artifact. |
| Required capabilities | Periodic Cartesian Poisson, FFT or GeometricMG, constant-kernel gauge. MPI off. |
| Configurations | Uniform 1-d cells. Public Case: `-laplacian(phi)==rhs`, `FFT()`, `Periodic`, `ConstantNullspace`, `MeanValueGauge(0)`. |
| Diagnostics | Task 2 L1/L2/L∞ of manufactured \(\phi\) and exact \(E\). Task 15 observed order of \(E\propto h^2\). Residual of \(\rho-(2\pi)^2\phi\). |
| Thresholds | Spatial order threshold \(1.8\) for a declared second-order scheme. Identity residual is 0. |
| Proves | 1-d oracle identities, manufactured second-order observed order, and a schema-valid campaign report without a solver. |
| Does not prove | Native elliptic solve, 2-d/3-d Poisson, AMR, coupling, MPI, performance. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
