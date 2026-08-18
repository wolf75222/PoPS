# CP-02 — cold Langmuir wave

Phase 2 1-d closed-form cold Langmuir standing wave. Units
\(e=m_e=\varepsilon_0=1\), \(n_0=1\) so \(\omega_{pe}=1\). Native compile is
optional; the in-memory path checks Gauss and the exact \(E(t)\) probe.

| Field | Content |
|---|---|
| Identifier | `CP-02` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | Cold electrons: \(\partial_t n_e+\partial_x(n_e u_e)=0\), \(\partial_t(n_e u_e)+\partial_x(n_e u_e^2)=(q_e/m_e)n_e E\) with \(q_e=-e\). Fixed ions \(n_i=n_0\). Gauss \(\partial_x E=e(n_i-n_e)/\varepsilon_0\). \(E=-\partial_x\phi\). |
| Oracle | Closed 1-d standing wave \(n_e=n_0+A\cos(kx)\cos(\omega_{pe}t)\), \(u_e=(A\omega_{pe})/(n_0 k)\sin(kx)\sin(\omega_{pe}t)\), \(E=-(e A)/(\varepsilon_0 k)\sin(kx)\cos(\omega_{pe}t)\), \(\phi=-(e A)/(\varepsilon_0 k^2)\cos(kx)\cos(\omega_{pe}t)\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). |
| Parameters | Dimensionless. \(e=m_e=\varepsilon_0=n_0=n_i=1\Rightarrow\omega_{pe}=1\). \(k=2\pi\). \(A=10^{-4}\). |
| Native dimensions | `POPS_NATIVE_DIM=1`. In-memory path does not load a native artifact. |
| Required capabilities | Cartesian 1-d, uniform, periodic, Poisson, electrostatic source, KokkosSerial, MPI off. |
| Configurations | Single resolution \(n=64\) for the in-memory report. Public Case: pressureless Rusanov, MUSCL/VanLeer, FFT Poisson, \(q_e n E/m_e\) source. No field-at-stage Program (resolve is `AuthoringPending`). |
| Diagnostics | Task 17 `numerical_frequency` / `frequency_error` on the exact \(E(t)\) probe. Spectral Gauss residual \(\partial_x E-e(n_i-n_e)/\varepsilon_0\). Task 2 exact-vs-exact L∞ on \(E\) and \(\phi\). |
| Thresholds | Exact-vs-exact L∞ = 0. Gauss residual L2 = 0. \(E_\omega=0\) on the exact probe. In-memory `coupling.sign_ok` is true when Gauss and \(\omega_{pe}\) identities hold. Do not flip the Poisson sign. |
| Proves | Closed 1-d Langmuir oracle satisfies Gauss; exact \(E(t)\) recovers \(\omega_{pe}=1\); report renderer. |
| Does not prove | Native frequency order, warm dispersion, AMR, MPI, 2-d oblique waves, energy exchange, eigenmode generator CP-05. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
