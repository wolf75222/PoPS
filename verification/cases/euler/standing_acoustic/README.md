# EU-04 — Standing acoustic wave, reflecting walls

1-d linear acoustic standing wave in a closed cavity. Oracle is the
linearized reflecting-wall solution. Native compile is optional.

| Field | Content |
|---|---|
| Identifier | `EU-04` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | 1-d gamma-law Euler, primitives \(W=(\rho,u,p)\). Conserved \((\rho,\rho u,E)\). No sources. \(\gamma=1.4\). |
| Oracle | Linear standing wave \(\rho=\bar\rho+\varepsilon\cos(k\pi x)\cos(\omega t)\), \(u=(\varepsilon c/\bar\rho)\sin(k\pi x)\sin(\omega t)\), \(p=\bar p+\varepsilon c^2\cos(k\pi x)\cos(\omega t)\), \(\omega=k\pi c\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d cavity \([0,1]\). Walls at \(x=0,1\) with \(u=0\). Authored in-memory layout uses the periodic helper and is not run. |
| Parameters | \(k=1\), \(\bar\rho=1\), \(\bar u=0\), \(\bar p=1/\gamma\) so \(c=1\). \(\varepsilon=10^{-3}\). Period \(2/c\). Dimensionless. |
| Native dimensions | `POPS_NATIVE_DIM=1`. In-memory path does not load a native artifact. |
| Required capabilities | Cartesian 1-d, uniform, reflecting walls, KokkosSerial, MPI off. HLLC/Rusanov + MUSCL when a native series exists. |
| Configurations | Single resolution \(n=32\) for the in-memory report. Authored scheme: Rusanov, MUSCL/VanLeer, SSPRK2, AdaptiveCFL. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ on density of exact vs exact. Wall velocity. Acoustic-energy period. Density/pressure phase. |
| Thresholds | Exact-vs-exact L∞ = 0. No spatial-order gate on the in-memory path. |
| Proves | Reflecting-wall standing-wave oracle (\(u=0\) at the walls; energy period \(2/c\); \(\rho\) and \(p\) in phase); report renderer. |
| Does not prove | Observed spatial/temporal order, wall-BC implementation, AMR, Poisson, coupling, MPI, HLLC vs Rusanov parity. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
