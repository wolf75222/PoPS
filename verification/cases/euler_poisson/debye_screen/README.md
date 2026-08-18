# CP-09 — linearized Debye screen

Phase 2 Euler–Poisson identity. One-dimensional periodic Helmholtz
\((-d^2/dx^2+\lambda_D^{-2})\phi=f\) with a cosine source. Native compile
is optional.

| Field | Content |
|---|---|
| Identifier | `CP-09` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | Periodic Helmholtz \((-\partial_{xx}+\lambda_D^{-2})\phi=f\). Manufactured \(f=\cos(2\pi k x)\), \(\phi=f/((2\pi k)^2+\lambda_D^{-2})\). \(E=-\partial_x\phi\). As \(\lambda_D\to\infty\), \(\phi\to f/(2\pi k)^2\) (Poisson). |
| Oracle | \(\lambda_D=0.1\), \(k=1\). `f_exact`, `phi_exact`, spectral `apply_helmholtz`. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Samples are uniform cell centers. |
| Parameters | Dimensionless. Default \(N=32\). Screening length \(\lambda_D=0.1\). Integer mode \(k=1\). Poisson-limit probe \(\lambda_D=10^8\). |
| Native dimensions | `POPS_NATIVE_DIM=1`. In-memory path does not load a native artifact. |
| Required capabilities | Periodic Cartesian Helmholtz / screened Poisson, FFT. MPI off. Invertible reaction term; no constant-kernel gauge. |
| Configurations | Uniform 1-d cells. Public Case: `-laplacian(phi)+λ_D^{-2} phi==f`, `FFT()`, `Periodic`. No AMR. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ of Helmholtz residual vs 0, \(\phi\) vs closed form, \(E\) vs closed form. Poisson-limit gain \(1/(2\pi k)^2\). |
| Thresholds | Exact-vs-exact L∞ on \(\phi\) and \(E\) is 0. Residual L2 is 0. Large-\(\lambda_D\) L∞ vs Poisson \(\le 10^{-10}\). No spatial-order gate on the in-memory path. |
| Proves | Analytic Helmholtz identity; \(\lambda_D\to\infty\) recovers Poisson \(1/(2\pi k)^2\); report renderer. |
| Does not prove | Native elliptic solve, live two-fluid Euler–Poisson coupling, nonlinear Debye, AMR, MPI, performance. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
