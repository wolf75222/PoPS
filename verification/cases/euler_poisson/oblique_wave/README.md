# CP-04 — oblique electrostatic eigenmode

Phase 4 2-d closed-form cold Langmuir-style mode with integer wavevector
\(k=(1,2)\) on the unit square. Native compile is optional; the in-memory
path checks Poisson and the \((k_x,k_y)\leftrightarrow(k_y,k_x)\) permutation.

| Field | Content |
|---|---|
| Identifier | `CP-04` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | Cold electrons on the unit square. Gauss \(\nabla\cdot E=e(n_i-n_e)/\varepsilon_0\), \(E=-\nabla\phi\). Fourier form \(iK\cdot\hat E=\hat\rho/\varepsilon_0\) with \(K=2\pi k\). Fixed ions \(n_i=\bar n\). |
| Oracle | Simplest consistent pack: \(\phi=\varepsilon\cos(2\pi(k_x x+k_y y)-\omega t)\), \(E=-\nabla\phi\), \(\delta n=(\varepsilon_0/e)\Delta\phi\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 2-d periodic unit square \([0,1]^2\). 1-d is not applicable. |
| Parameters | Dimensionless. \(e=m_e=\varepsilon_0=\bar n=n_i=1\Rightarrow\omega_{pe}=1\). \(k=(1,2)\). \(\varepsilon=10^{-4}\) (potential amplitude). \(\omega\) is a free parameter (default \(\omega_{pe}\)). |
| Native dimensions | `POPS_NATIVE_DIM=2`. In-memory path does not load a native artifact. |
| Required capabilities | Cartesian 2-d, uniform, periodic, Poisson, electrostatic source, KokkosSerial, MPI off. |
| Configurations | Single resolution \(n=32^2\) for the in-memory report. Public Case: pressureless Rusanov, MUSCL/VanLeer, SSPRK2. Poisson stays on the oracle. |
| Diagnostics | Fourier Poisson identity \(iK\cdot\hat E=\hat\rho/\varepsilon_0\). Analytic Gauss residual \(\nabla\cdot E-e(n_i-n_e)/\varepsilon_0\). Coordinate permutation under \((k_x,k_y)\leftrightarrow(k_y,k_x)\). Task 2 exact-vs-exact L∞ on \(n\), \(\phi\), and \(E\). |
| Thresholds | Exact-vs-exact L∞ = 0. Gauss residual L2 = 0. Fourier residual \(\le 10^{-14}\). In-memory `coupling.sign_ok` is true when Gauss, \(iK\cdot\hat E\), and the swap permutation hold. Do not flip the Poisson sign. |
| Proves | Closed 2-d oblique pack satisfies Poisson; \(k\) is not axis-aligned; swapping \((k_x,k_y)\) is a coordinate permutation of the same field; report renderer. |
| Does not prove | Native frequency order, warm dispersion, AMR, MPI, 1-d reduction, energy exchange, eigenmode generator CP-05. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
