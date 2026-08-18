# PO-04 — Huang–Greengard multi-blob Poisson (1-d stand-in)

Phase 1 code-verification case. In-memory manufactured 1-d periodic
multi-Gaussian Poisson only. Public Case authoring is present. No native
compile, no bind, no `pops.run`, no ROMEO.

| Field | Content |
|---|---|
| Identifier | `PO-04` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | \(-\phi''=\rho\) on a 1-d periodic interval. Manufactured \(\phi=\sum_i A_i\exp(-((x-c_i)/\sigma)^2)\) with centres \(c=(0.25,0.5,0.75)\), \(A_i=1\), \(\sigma=0.04\). Displacement is the nearest periodic image. \(\rho=-\phi''\) is analytic. \(E=-d\phi/dx\). |
| Oracle | `phi_exact` / `rhs_exact` / `dphi_exact` / `d2phi_exact` / `e_exact` / `nearest_image_displacement` in `exact.py`. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Samples are uniform cell centers. |
| Parameters | Dimensionless. \(\sigma=0.04\). Manufactured order series \(N=16,32,64,128\). Optional \(C h^2\) perturbation of \(\phi\). Finite-difference residual series \(N=128,256,512\). |
| Native dimensions | `POPS_NATIVE_DIM=1` only. This worktree does not load a native artifact. |
| Required capabilities | Periodic Cartesian Poisson, FFT or GeometricMG, constant-kernel gauge. MPI off. |
| Configurations | Uniform 1-d cells. Public Case: `-laplacian(phi)==rhs`, `FFT()`, `Periodic`, `ConstantNullspace`, `MeanValueGauge(0)`. |
| Diagnostics | Analytic \(\rho+\phi''=0\). Periodic second-difference residual \(\|-D^2\phi-\rho\|_\infty\to 0\) as \(N\) increases. Localization: \(\lvert\phi\rvert>10^{-6}\) only within \(5\sigma\) of a centre. Task 2 L1/L2/L∞ of manufactured \(\phi\) and exact \(E\). Task 15 observed order of \(E\propto h^2\). |
| Thresholds | Spatial order threshold \(1.8\) for a declared second-order scheme. Analytic identity residual is 0. Finite-difference residual order \(>1.5\) on \(N=128,256,512\). |
| Proves | Nearest-image multi-blob oracle, analytic \(-\phi''=\rho\), finite-difference residual decreases under refinement, blobs are localized, manufactured second-order observed order, and a schema-valid campaign report without a solver. |
| Does not prove | Native elliptic solve, 2-d/3-d Huang–Greengard, FMM, AMR, coupling, MPI, performance. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
