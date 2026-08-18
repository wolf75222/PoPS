# EU-05 — Gresho vortex (2-d, stationary)

Standard 2-d compressible Gresho vortex. The exact field is a stationary
centrifugal equilibrium: piecewise \(u_\theta(r)\) with \(p(r)\) integrated so
\(\nabla p=\rho u_\theta^2/r\). 1-d is not applicable. Native compile is optional.

| Field | Content |
|---|---|
| Identifier | `EU-05` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | 2-d gamma-law Euler, primitives \(W=(\rho,u,v,p)\). Conserved \((\rho,\rho u,\rho v,E)\). No sources. \(\gamma=1.4\). |
| Oracle | Stationary Gresho vortex. \(\rho=1\). \(u_\theta(r)=5r\) (\(r<0.2\)), \(2-5r\) (\(0.2\le r<0.4\)), \(0\) (\(r\ge 0.4\)). Three-piece pressure from \(\mathrm{d}p/\mathrm{d}r=\rho u_\theta^2/r\): \(p=5+12.5 r^2\) (\(r<0.2\)); \(p=9+12.5 r^2-20 r+4\ln(5r)\) (\(0.2\le r<0.4\)); \(p=3+4\ln 2\) (\(r\ge 0.4\)). Cartesian \((u,v)=(-u_\theta\sin\theta,u_\theta\cos\theta)\). Independent of \(t\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 2-d periodic box \([0,1]^2\), centre \((0.5,0.5)\). 1-d is not applicable. |
| Parameters | \(\rho=1\), \(\gamma=1.4\), \(p(0)=5\), kinks at \(r=0.2\) and \(r=0.4\). Dimensionless. |
| Native dimensions | `POPS_NATIVE_DIM=2`. In-memory path does not load a native artifact. |
| Required capabilities | Cartesian 2-d, uniform, periodic, KokkosSerial, MPI off. HLLC/Rusanov + MUSCL when a native series exists. |
| Configurations | Single resolution \(n=32^2\) for the in-memory report. Authored scheme: Rusanov, MUSCL/VanLeer, SSPRK2, AdaptiveCFL. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ on density of exact vs exact. Radial velocity identically 0. Pressure continuous at \(r=0.2,0.4\). Centrifugal residual \(\mathrm{d}p/\mathrm{d}r-\rho u_\theta^2/r\) near 0 inside each piece. |
| Thresholds | Exact-vs-exact L∞ = 0. Empty `orders` with reason containing `stationary vortex`. Residual atol \(10^{-12}\). |
| Proves | Gresho oracle (purely azimuthal velocity; \(C^0\) pressure at the kinks; centrifugal balance of the three-piece \(p(r)\)); report renderer accepts empty orders justified by a stationary vortex. |
| Does not prove | Observed spatial/temporal order, dissipation of a numerical Gresho, AMR, Poisson, coupling, MPI, HLLC vs Rusanov parity, 1-d reduction. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
