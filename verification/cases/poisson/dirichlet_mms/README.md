# PO-02 — Dirichlet MMS Poisson (2-d)

Phase 1 code-verification case. In-memory manufactured 2-d Dirichlet
Poisson only. Public Case authoring is present. No native compile, no bind,
no `pops.run`, no ROMEO.

| Field | Content |
|---|---|
| Identifier | `PO-02` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | \(-\Delta\phi=f\) on the unit square. Manufactured \(\phi=e^{x}\sin(2\pi y)+x^{2}y\), \(f=(4\pi^{2}-1)e^{x}\sin(2\pi y)-2y\), \(\mathbf{E}=-\nabla\phi\). 1-d slice \(\phi=e^{x}\), \(f=-e^{x}\). |
| Oracle | `phi_exact` / `rhs_exact` / `e_exact` in `exact.py`. 1-d slice `phi_exact_1d` / `rhs_exact_1d` / `e_exact_1d`. `exact.py` does not read PoPS output. |
| Domain and boundaries | Unit square \([0,1]^{2}\) with Dirichlet data equal to the exact \(\phi\) on \(\partial\Omega\). Samples are uniform cell centers. 1-d slice on \([0,1]\). |
| Parameters | Dimensionless. Manufactured order series \(N=16,32,64,128\). Optional \(C h^{2}\) perturbation of \(\phi\). |
| Native dimensions | `POPS_NATIVE_DIM=2` primary. 1-d slice is exported for authoring. This worktree does not load a native artifact. |
| Required capabilities | Dirichlet Cartesian Poisson, GeometricMG (FFT is periodic-only). MPI off. |
| Configurations | Uniform 2-d cells. Public Case: `-laplacian(phi)==rhs`, `GeometricMG()`, `Dirichlet`. |
| Diagnostics | Task 2 L1/L2/L∞ of manufactured \(\phi\) and exact \(\|\mathbf{E}\|\). Task 15 observed order of \(E\propto h^{2}\). Residual of \(f+\Delta\phi\). |
| Thresholds | Spatial order threshold \(1.8\) for a declared second-order scheme. Identity residual is 0. |
| Proves | 2-d \(\Delta\) identity, 1-d slice identity, manufactured second-order observed order, and a schema-valid campaign report without a solver. |
| Does not prove | Native elliptic solve, inhomogeneous Dirichlet enforcement at runtime, AMR, coupling, MPI, performance. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
