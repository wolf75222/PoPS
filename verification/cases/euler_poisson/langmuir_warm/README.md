# CP-03 — warm Langmuir dispersion

1-d warm Langmuir linear dispersion of an electron fluid about a uniform
equilibrium with fixed ions. Oracle is \(\omega^2=\omega_{pe}^2+c_e^2k^2\).
Native compile is optional.

| Field | Content |
|---|---|
| Identifier | `CP-03` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | Linearized electron fluid + Poisson. Continuity \(\partial_t n_1+n_0\partial_x u=0\). Momentum \(\partial_t u+(c_e^2/n_0)\partial_x n_1=(q_e/m_e)E\). Gauss \(\partial_x E=e(n_i-n_e)/\varepsilon_0\). Units \(e=m_e=\varepsilon_0=1\), \(n_0=1\Rightarrow\omega_{pe}=1\), \(q_e=-e\). |
| Oracle | Dispersion \(\omega^2=\omega_{pe}^2+c_e^2k^2\). Closed eigenmode \(n_e=n_0+A\cos(kx)\cos(\omega t)\), \(u_e=(A\omega)/(n_0k)\sin(kx)\sin(\omega t)\), \(E=-(eA)/(\varepsilon_0k)\sin(kx)\cos(\omega t)\), \(\phi=-(eA)/(\varepsilon_0k^2)\cos(kx)\cos(\omega t)\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). |
| Parameters | \(\omega_{pe}=1\), \(c_e=0.2\), \(A=10^{-4}\). Sweep \(k/2\pi=1,2,4,8\). Dimensionless. |
| Native dimensions | `POPS_NATIVE_DIM=1`. In-memory path does not load a native artifact. |
| Required capabilities | Cartesian 1-d, uniform, periodic, KokkosSerial, MPI off. |
| Configurations | Single resolution \(n=32\) for the in-memory report. Authored scheme: Rusanov, MUSCL/VanLeer, SSPRK2, AdaptiveCFL. Poisson coupling stays on the oracle. |
| Diagnostics | Dispersion residual \(\omega^2-(\omega_{pe}^2+c_e^2k^2)\) at the four wavenumbers. Task 2 volume-weighted L1/L2/L∞ on density of exact vs exact. |
| Thresholds | Dispersion residual \(\le 10^{-12}\). Exact-vs-exact L∞ = 0. No spatial-order gate on the in-memory path. |
| Proves | Warm Langmuir dispersion identity at \(k/2\pi=1,2,4,8\); report renderer. |
| Does not prove | Native \(\omega_{\mathrm{num}}(k)\), numerical damping, modal contamination, observed spatial/temporal order, AMR, MPI, oblique 2-d/3-d waves, kinetic Landau damping. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
