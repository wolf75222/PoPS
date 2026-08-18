# CP-01 — Euler–Poisson manufactured solution

1-d canonical electrostatic MMS. The potential is prescribed; electron
density follows the closed Poisson identity. Ions are fixed. Native
compile is optional.

**Poisson sign (do not flip in `analyze.py`).** Canonical convention
from plan §14:

\[
-\varepsilon_0\,\partial_{xx}\phi=e(n_i-n_e),\qquad
E=-\partial_x\phi,\qquad q_e=-e.
\]

With \(\phi=A\cos(kx-\omega t)\) this is equivalent to
\(n_i-n_e=\varepsilon_0 k^2\phi/e\). `analyze.py` evaluates the residual
\(-\varepsilon_0\phi_{xx}-e(n_i-n_e)\) with that sign. A PoPS elliptic
operator that uses another convention must be transformed in the README
and Case, never by flipping the residual after the fact.

| Field | Content |
|---|---|
| Identifier | `CP-01` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | 1-d Euler–Poisson. Electrons: primitives \(W=(n_e,u_e,p_e)\), conserved \((n_e,n_e u_e,\mathcal{E})\), \(\gamma=5/3\), \(m_e=1\), \(q_e=-e\). Ions fixed at \(n_i\). Poisson \(-\varepsilon_0\phi_{xx}=e(n_i-n_e)\). Manufactured hyperbolic source \(S=\partial_t U+\partial_x F-q_e n_e E\,(0,1,u_e)/m_e\). No manufactured source in Poisson. |
| Oracle | \(\phi=A\cos(kx-\omega t)\), \(n_e=n_i-(\varepsilon_0 k^2/e)\phi\), \(E=Ak\sin(kx-\omega t)\), \(u_e=0.5+0.1\cos(kx-\omega t)\), \(p_e=1+0.05\cos(kx-\omega t)\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Samples are uniform cell centers. |
| Parameters | \(e=1\), \(\varepsilon_0=1\), \(n_i=1\), \(A=10^{-3}\), \(k=2\pi\), \(\omega=2\pi\), \(\gamma=5/3\). Then \(n_e\in[1-4\pi^2\cdot10^{-3},1+4\pi^2\cdot10^{-3}]\subset(0.96,1.04)\). \(u_e\in[0.4,0.6]\), \(p_e\in[0.95,1.05]\). |
| Native dimensions | `POPS_NATIVE_DIM=1`. In-memory path does not load a native artifact. |
| Required capabilities | Cartesian 1-d, uniform, periodic, KokkosSerial, MPI off. Coupled Euler–Poisson when a native series exists. |
| Configurations | Single resolution \(n=32\) for the in-memory report. Authored scheme: Rusanov, MUSCL/VanLeer, SSPRK2, AdaptiveCFL. Hyperbolic sources are exported by `exact.sources_1d` and are not yet injected into the Case. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ on \(n_e\), \(\phi\), \(E\) of exact vs exact. Poisson residual with the documented sign. Coupling `sign_ok` from \(n_i-n_e=\varepsilon_0 k^2\phi/e\). Positivity of \(n_e,u_e,p_e\). |
| Thresholds | Exact-vs-exact L∞ = 0. Documented Poisson residual L2 = 0. No spatial-order gate on the in-memory path. |
| Proves | Closed 1-d charge–potential identity; positivity; documented Poisson sign without post-hoc flip; report renderer. |
| Does not prove | Observed spatial/temporal order, native coupled solve, field at every RK stage, AMR, MPI, Dirichlet, two-species, 2-d. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
