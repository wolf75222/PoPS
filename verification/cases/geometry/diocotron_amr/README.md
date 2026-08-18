# GE-06 — Cartesian diocotron + AMR companion

Cartesian stand-in of the CP-11 hollow density ring, tagged for a two-level
AMR envelope. Public Case is 2-d pressureless Euler–Poisson on a uniform
Cartesian mesh. Polar is refused.

| Field | Content |
|---|---|
| Identifier | `GE-06` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | Number density on a 2-d Cartesian mesh. Unperturbed ring \(n(r)=n_0\) for \(r_1<r<r_2\), else \(n_{\mathrm{bg}}\). Tag \(\lvert n-n_{\mathrm{bg}}\rvert>\theta\). Two-level envelope = tagged cells plus Chebyshev buffer 2. Optional CP-11 azimuthal seed \(\varepsilon\Re(e^{im\theta}e^{\gamma t})\) with toy \(\gamma=0.1\) is not used on the unperturbed path. Native smoke: cold-electron Euler–Poisson, \(e=m_e=\varepsilon_0=1\), neutralizing \(n_i=\langle n\rangle\), \(q_e=-e\). |
| Oracle | CP-11 ring via `load_sibling_module` on `verification/cases/euler_poisson/diocotron/exact.py` when present; otherwise the same formula is duplicated in this `exact.py`. CP-11 `density` is **not** sampled on the GE-06 origin-centered mesh (that sibling is unit-square centered at 0.5). `exact.py` does not read PoPS output. |
| Domain and boundaries | 2-d box \([-1/2,1/2]^2\), ring centred at the origin. Periodic uniform layout. The ring stays away from the boundary (\(r_2=0.35\)). Cartesian only. |
| Parameters | Dimensionless. \(n_0=1\), \(n_{\mathrm{bg}}=0\), \(r_1=0.20\), \(r_2=0.35\), \(\theta=0.5\), buffer \(=2\). Unused mode \(m=3\). Default in-memory grid \(n=32^2\). Native IC floors \(n\) at \(10^{-8}\) so rest-fluid waves stay defined. |
| Native dimensions | `POPS_NATIVE_DIM=2`. `run_native` compiles a uniform Dim2 Euler–Poisson Case (`fields=`, `model.aux("potential")` / `phi_grad_x` / `phi_grad_y`, FFT, `FixedDt`). Polar System is refused. |
| Required capabilities | Cartesian 2-d, uniform, KokkosSerial, MPI off. A later native series needs 2-d tagging, buffer dilation, and a two-level envelope. |
| Configurations | In-memory report at \(n=32^2\). Authored scheme: pressureless Euler–Poisson, Rusanov, MUSCL/VanLeer, SSPRK2(`fields=`), `FixedDt`. No live regrid. |
| Diagnostics | Raw tag mask vs ring; Chebyshev buffer-2 envelope; unused-mode (\(m=3\)) angular rFFT of the unperturbed ring on \(r=(r_1+r_2)/2\). Task 2 exact-vs-exact L1/L2/L∞ on \(n\). |
| Thresholds | Tagged set covers the ring. Envelope is tagged plus buffer 2. \(\lvert\mathrm{FFT}[m=3]\rvert\approx 0\) (atol \(10^{-12}\)). Exact-vs-exact L∞ = 0. No spatial-order gate on the in-memory path. |
| Proves | \(\lvert n-n_{\mathrm{bg}}\rvert>\theta\) tags the CP-11 ring; the two-level envelope is that mask plus a two-cell halo; the unperturbed ring has no unused \(m=3\) content; public 2-d Cartesian Euler–Poisson resolves; report renderer. |
| Does not prove | Live AMR regrid, clustering, order retention, linear diocotron growth (CP-11), coupling, MPI, native Cartesian AMR. Public AM-01/AM-11 contracts are 1-d marker/tag layouts; a live 2-d \(\lvert n-n_{\mathrm{bg}}\rvert\) tag is not a copyable public rule, so the two-level envelope stays the in-memory oracle. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compile is optional (`run_native`). |

## AMR gap

AM-01 (static coarse/fine marker) and AM-11 (Euler–Poisson AMR, `fields=`,
`GeometricMG` + `CompositeHierarchySolve`) are public **1-d** contracts. Copying
them onto `Cartesian2D` would invent a 2-d tag (AM-11 tags a prescribed marker,
not \(\lvert n-n_{\mathrm{bg}}\rvert>\theta\)). GE-06 therefore keeps the
Chebyshev buffer-2 envelope as the in-memory oracle and runs a **uniform** Dim2
Euler–Poisson smoke.
