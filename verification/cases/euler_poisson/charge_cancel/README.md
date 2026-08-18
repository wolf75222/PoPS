# CP-12 — charge cancellation

Phase 2 Euler–Poisson identity. Two species share the same number density
and carry opposite charges, so the net charge vanishes and the electric
field is identically zero. Native compile is optional.

| Field | Content |
|---|---|
| Identifier | `CP-12` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | Two-species charge \(\rho_q=q n+(-q)n=0\). Periodic Poisson \(-\varepsilon_0\phi''=\rho_q\). Gauge \(\langle\phi\rangle=\mathrm{PHI0}=0\). \(E=-d\phi/dx\). |
| Oracle | Shared \(n=n_0+\delta\cos(2\pi x)\) for both species, \(q=1\), \(n_0=1\), \(\varepsilon_0=1\). Then \(\rho_q=0\), \(\phi=\mathrm{PHI0}\), \(E=0\). Species accumulation order does not change \(\rho_q\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Samples are uniform cell centers. |
| Parameters | Dimensionless. Default \(N=32\). Uniform \(\delta=0\) and compensating \(\delta=0.1\). |
| Native dimensions | `POPS_NATIVE_DIM=1`. In-memory path does not load a native artifact. |
| Required capabilities | Periodic Cartesian Poisson, FFT, constant-kernel gauge. MPI off. Two-species charge assembly. |
| Configurations | Uniform 1-d cells. Public Case: `-laplacian(phi)==rhs`, `FFT()`, `Periodic`, `ConstantNullspace`, `MeanValueGauge(0)`. No AMR. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ of net charge vs 0, \(E\) vs 0, \(\phi\) vs \(\mathrm{PHI0}\), Poisson residual vs 0. |
| Thresholds | Exact-vs-exact L∞ on charge, \(E\), and \(\phi\) is 0. Residual L2 is 0. No spatial-order gate on the in-memory path. |
| Proves | Analytic two-species cancellation \(qn+(-q)n=0\); \(E=0\) and constant \(\phi\) after the mean-zero gauge; species-order independence of \(\rho_q\); report renderer. |
| Does not prove | Native elliptic solve, live two-fluid Euler–Poisson coupling, three-species or mass-ratio variants, AMR, MPI, performance, parasitic-momentum evolution. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
