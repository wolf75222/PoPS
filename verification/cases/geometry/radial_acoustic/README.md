# GE-03 — Radial acoustic wave in Cartesian (2-d)

Small-amplitude radial standing wave on a 2-d Cartesian box. The exact field
depends only on \(r=\sqrt{x^2+y^2}\). Public 2-d Euler authoring plus a Dim2
`run_native` smoke. No polar mesh.

| Field | Content |
|---|---|
| Identifier | `GE-03` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | 2-d linear acoustics on a gamma-law Euler background. Velocity potential \(\psi=\varphi\). \(\mathbf{u}=\nabla\psi\), \(p'=-\rho_0\partial_t\psi\), \(\rho'=p'/c^2\). \(\gamma=1.4\). |
| Oracle | \(\varphi=\varepsilon J_0(k r)\cos(\omega t)\) with \(\omega=c k\), \(c=1\), \(\varepsilon=10^{-3}\), \(k=1\). \(J_0\) from its power series. `exact.py` does not read PoPS output. |
| Domain and boundaries | 2-d box \([-1,1]^2\), wave centred at the origin. Periodic `uniform_periodic_layout`. |
| Parameters | Dimensionless. \(\rho_0=1\), \(p_0=1/\gamma\) so \(c=\sqrt{\gamma p_0/\rho_0}=1\). Default in-memory grid \(n=32^2\). |
| Native dimensions | `POPS_NATIVE_DIM=2`. `run_native` refuses any other value (no fallback). |
| Required capabilities | Cartesian 2-d, uniform, KokkosSerial, MPI off. A later native series needs 2-d Euler. |
| Configurations | Single resolution \(n=32^2\) for the in-memory report. Authored scheme: Rusanov, MUSCL/VanLeer, SSPRK2, AdaptiveCFL. |
| Diagnostics | Angular standard deviation of \(\varphi\) on the circle \(r=0.5\) at \(t=0\). Dispersion \(\omega=c k\). Task 2 exact-vs-exact L1/L2/L∞ on \(\varphi\). |
| Thresholds | Angular std \(\approx 0\) (atol \(10^{-12}\)). \(\omega=c k\) exactly. Exact-vs-exact L∞ = 0. No spatial-order gate on the in-memory path. |
| Proves | Cartesian samples of the Bessel standing wave depend only on \(r\); linear-acoustic primitives are consistent with \(\mathbf{u}=\nabla\psi\); \(\omega=c k\); report renderer. |
| Does not prove | Observed spatial/temporal order, polar-mesh equivalence, AMR, Poisson, coupling, MPI, native Cartesian isotropy. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
