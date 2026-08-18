# CP-05 — multifluid eigenmode generator

2×2 toy linearized generator. Fourier symbol \(M(k)\) has known eigenpairs.
The closed reference is \(U=\bar U+\epsilon\Re(r\exp(ikx+\lambda t))\). Public
1-d periodic authoring uses the acoustic flux \((c u, c n)\) with waves
\(\pm c\). Native compile is optional.

| Field | Content |
|---|---|
| Identifier | `CP-05` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | Two-component acoustic toy \(\partial_t n+c\partial_x u=0\), \(\partial_t u+c\partial_x n=0\). Fourier: \(\partial_t\widehat U=M(k)\widehat U\) with \(M(k)=\begin{pmatrix}0&-ick\\-ick&0\end{pmatrix}\). |
| Oracle | Known eigenpairs \(\lambda_\pm=\pm ick\), \(r_+=(1,-1)\), \(r_-=(1,1)\). Closed field \(U(x,t)=\bar U+\epsilon\Re(r\exp(ikx+\lambda t))\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). |
| Parameters | \(c=1\), \(\bar U=(1,0)\), \(\epsilon=10^{-4}\), \(k=2\pi\). Dimensionless. |
| Native dimensions | `POPS_NATIVE_DIM=1`. `run_native` returns the acoustic state as shape `(2, n)`. |
| Required capabilities | Cartesian 1-d, uniform, periodic, KokkosSerial, MPI off. Rusanov + MUSCL when a native series exists. |
| Configurations | Single resolution \(n=32\) for the in-memory report. Authored scheme: Rusanov, MUSCL/VanLeer, SSPRK2, AdaptiveCFL. BindArray IC from `exact_state(..., t=0, mode=)`. No Poisson. |
| Diagnostics | Eigenvector residual \(\|Mr-\lambda r\|_\infty\). Closed-form time advance of each mode, including \(\exp(Mt)\hat U\). Task 2 volume-weighted L1/L2/L∞ on density of exact vs exact. |
| Thresholds | Eigenvector residual \(\le 10^{-12}\). Time-advance mismatch \(\le 10^{-12}\). Exact-vs-exact L∞ = 0. No spatial-order gate on the in-memory path. |
| Proves | 2×2 \(M(k)\) eigenpairs; closed time advance \(e^{\lambda t}\); report renderer; public 1-d periodic Case validates and resolves. |
| Does not prove | Native multi-fluid Euler–Poisson, ion-acoustic / Langmuir / collisional / magnetized modes, modal contamination order, AMR, MPI, DSL/native parity. Native compile requires Kokkos + a compiler. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
