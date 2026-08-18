# PO-03 — Neumann nullspace Poisson (1-d)

Phase 1 code-verification case. In-memory manufactured 1-d homogeneous-Neumann
Poisson only. Public Case authoring is present. No native compile, no bind,
no `pops.run`, no ROMEO.

| Field | Content |
|---|---|
| Identifier | `PO-03` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | \(-\phi''=\rho\) on a 1-d interval with homogeneous Neumann \(\phi'=0\) at both ends. Manufactured \(\phi=\cos(2\pi x)\), \(\rho=(2\pi)^2\phi\), \(E=-d\phi/dx=2\pi\sin(2\pi x)\). Constant nullspace is gauged by comparing \(\phi-\langle\phi\rangle\). |
| Oracle | `phi_exact` / `rhs_exact` / `dphi_exact` / `e_exact` / `mean_free` in `exact.py`. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d unit interval \([0,1]\). Homogeneous Neumann at \(x=0\) and \(x=1\). Samples are uniform cell centers. |
| Parameters | Dimensionless. Manufactured order series \(N=16,32,64,128\). Optional constant gauge shift plus \(C h^2\phi\) perturbation of \(\phi\). |
| Native dimensions | `POPS_NATIVE_DIM=1` only. This worktree does not load a native artifact. |
| Required capabilities | Cartesian Neumann Poisson, GeometricMG, constant-kernel gauge. MPI off. |
| Configurations | Uniform 1-d cells. Public Case: `-laplacian(phi)==rhs`, `GeometricMG()`, `Neumann(0)`, `ConstantNullspace`, `MeanValueGauge(0)`. |
| Diagnostics | Task 2 L1/L2/L∞ of mean-free manufactured \(\phi\) and exact \(E\). Task 15 observed order of mean-free \(E\propto h^2\). Residual of \(\rho-(2\pi)^2\phi\). Compatibility of \(\int\rho\). |
| Thresholds | Spatial order threshold \(1.8\) for a declared second-order scheme. Identity residual is 0. Volume-weighted rhs mean must be ~0. |
| Proves | Homogeneous Neumann oracle (\(\phi'=0\) at the ends), mean-free comparison removes the constant nullspace, a constant rhs is incompatible, manufactured second-order observed order after subtracting \(\langle\phi\rangle\), and a schema-valid campaign report without a solver. |
| Does not prove | Native elliptic solve, 2-d/3-d Poisson, AMR, coupling, MPI, performance. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |

## Incompatible right-hand side

Homogeneous Neumann \(-\Delta\phi=f\) with \(\phi'=0\) on \(\partial\Omega\) is
solvable if and only if \(\int_\Omega f=0\). A constant (non-zero) rhs violates
that compatibility condition. `run.require_compatible_rhs` raises
`IncompatibleRhs`; `run.incompatible_rhs_observation` records the same fact
without raising. The manufactured \(\rho=(2\pi)^2\cos(2\pi x)\) is compatible.
