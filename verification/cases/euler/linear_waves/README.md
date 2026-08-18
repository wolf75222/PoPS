# EU-01 — Euler linear eigenmodes

1-d gamma-law Euler linear waves about a uniform rest state. Oracle is the
linearized eigenmode solution. Native compile is optional.

| Field | Content |
|---|---|
| Identifier | `EU-01` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | 1-d gamma-law Euler, primitives \(W=(\rho,u,p)\). Conserved \((\rho,\rho u,E)\). No sources. \(\gamma=1.4\). |
| Oracle | Linear eigenmode \(U(x,t)=\bar U+\varepsilon r_m\sin(kx-\lambda_m\|k\|t)\). Modes: left acoustic \(\lambda=u-c\), entropy \(\lambda=u\), right acoustic \(\lambda=u+c\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). |
| Parameters | \(\bar\rho=1\), \(\bar u=0\), \(\bar p=1/\gamma\) so \(c=1\). \(\varepsilon=10^{-6}\in[10^{-7},10^{-5}]\). \(k=2\pi\). Dimensionless. |
| Native dimensions | `POPS_NATIVE_DIM=1`. In-memory path does not load a native artifact. |
| Required capabilities | Cartesian 1-d, uniform, periodic, KokkosSerial, MPI off. HLLC/Rusanov + MUSCL when a native series exists. |
| Configurations | Single resolution \(n=32\) for the in-memory report. Authored scheme: Rusanov, MUSCL/VanLeer, SSPRK2, AdaptiveCFL. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ on one primitive (density) of exact vs exact. Eigenvector independence. Acoustic travel at \(+c\). |
| Thresholds | Exact-vs-exact L∞ = 0. No spatial-order gate on the in-memory path. |
| Proves | Linear eigenmode oracle (entropy stationary at \(u=0\); right acoustic translates at \(+c\)); report renderer; primitive eigenvectors are independent. |
| Does not prove | Observed spatial/temporal order, modal contamination, AMR, Poisson, coupling, MPI, HLLC vs Rusanov parity, multi-d oblique waves. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
