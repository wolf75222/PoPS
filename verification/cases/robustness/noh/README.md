# RB-06 — Noh planar

Classic 1-d planar Noh problem. Oracle is the exact self-similar
infinite-Mach solution. Native compile is optional.

| Field | Content |
|---|---|
| Identifier | `RB-06` |
| `verification_kind` | `robustness` |
| `evidence_status` | `required` |
| Equations | 1-d gamma-law Euler, primitives \(W=(\rho,u,p)\). Conserved \((\rho,\rho u,E)\). No sources. \(\gamma=5/3\). |
| Oracle | Self-similar planar Noh. Inflow toward \(x=0\): \((\rho,u,p)=(1,-\mathrm{sign}(x),0)\). For \(t>0\), shocks at \(\lvert x\rvert=t/3\); post-shock \(\rho=4\), \(u=0\), \(p=4/3\). Density ratio \((\gamma+1)/(\gamma-1)=4\); shock speed \(\lvert u\rvert(\gamma-1)/2=1/3\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d interval \([-1,1]\). Physical Noh needs inflow (or a wall at \(x=0\)); the authored in-memory layout uses the periodic helper and is not run. |
| Parameters | \(\gamma=5/3\). \(t=0.6\). Dimensionless. Oracle pre-shock \(p=0\) (cold gas). Native IC floors pressure at \(10^{-8}\) so cell-centered faces stay strictly positive; `exact.py` stays \(p=0\). |
| Native dimensions | `POPS_NATIVE_DIM=1`. In-memory path does not load a native artifact. |
| Required capabilities | Cartesian 1-d, uniform, KokkosSerial, MPI off. Rusanov + FirstOrder when a native series exists. |
| Configurations | Single resolution \(n=64\) for the in-memory report. Authored scheme: Rusanov, FirstOrder, SSPRK2, AdaptiveCFL, `FiniteVolume(positivity_floor=1e-8)`. MUSCL/VanLeer at the origin / periodic velocity jump reconstructs \(p<0\) on cold faces and publishes `InvalidWaveSpeed` (status=6) on the first residual. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ on density of exact vs exact. Post-shock \(\rho=4\). Shock position \(\lvert x\rvert=t/3\). Positivity of \(\rho\) and \(p\ge 0\). |
| Thresholds | Exact-vs-exact L∞ = 0. Empty `orders` with reason containing `shock / wall heating`. No spatial-order gate. |
| Proves | Exact self-similar Noh oracle (post-shock density 4; shock at \(t/3\); positivity); report renderer accepts empty orders justified by a shock / wall heating. |
| Does not prove | Observed spatial/temporal order, shock-capturing quality, wall-heating magnitude, AMR, Poisson, coupling, MPI, HLLC vs Rusanov parity, inflow/wall native BCs. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
