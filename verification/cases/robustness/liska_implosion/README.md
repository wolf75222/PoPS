# RB-07 — Liska–Wendroff implosion

Classic 2-d implosion (Liska & Wendroff 2003). Oracle is the diamond IC
plus leftover residual under reflection across \(x=y\). There is no
closed-form late-time solution. Public `SlipWall` on all faces plus a
Dim2 `run_native` smoke.

| Field | Content |
|---|---|
| Identifier | `RB-07` |
| `verification_kind` | `robustness` |
| `evidence_status` | `required` |
| Equations | 2-d gamma-law Euler, primitives \(W=(\rho,u,v,p)\). Conserved \((\rho,\rho u,\rho v,E)\). No sources. \(\gamma=1.4\). |
| Oracle | IC only. \((\rho,p)=(1,1)\) if \(x+y>0.15\), else \((0.125,0.14)\); \(u=v=0\). `reflect(q)` swaps \((u,v)\) (or \((\rho u,\rho v)\)) and transposes square 2-d fields. Leftover residual is \(q-\mathrm{reflect}(q)\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 2-d box \([0,0.3]^2\). Public `SlipWall` on every face; non-periodic `Uniform`. |
| Parameters | \(\gamma=1.4\). Cut \(x+y=0.15\). Morphological time \(t=2.5\) (not an oracle). Dimensionless. |
| Native dimensions | `POPS_NATIVE_DIM=2`. `run_native` refuses any other value (no fallback). |
| Required capabilities | Cartesian 2-d, uniform, KokkosSerial, MPI off. HLLC/Rusanov + MUSCL when a native series exists. Reflecting walls for a native run. |
| Configurations | Single resolution \(n=32\) for the in-memory report. Authored scheme: Rusanov, MUSCL/VanLeer, SSPRK2, AdaptiveCFL. |
| Diagnostics | IC jump on \(x+y=0.15\). Exact IC symmetric under \((x,y)\leftrightarrow(y,x)\). Leftover residual 0 on the exact field. Task 18 \(E_{xy}\) on density. Task 2 exact-vs-exact density norms. |
| Thresholds | Leftover residual of the exact IC is 0. Empty `orders` with reason containing `implosion / no analytic late-time`. No spatial-order gate. |
| Proves | Documented diamond IC; diagonal symmetry of the exact field; leftover residual vanishes on that field; report renderer accepts empty orders justified by a missing late-time closed form. |
| Does not prove | Observed spatial/temporal order, shock-capturing quality, jet/stem morphology at \(t=2.5\), AMR, Poisson, coupling, MPI. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
