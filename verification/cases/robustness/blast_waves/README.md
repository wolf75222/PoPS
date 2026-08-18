# RB-09 — Woodward–Colella blast waves

Standard 1-d two-blast IC on \([0,1]\) with reflecting walls. This
increment is the literature IC only. There is no closed-form evolved
state (interacting blasts) and no uniform-fine reference run.

| Field | Content |
|---|---|
| Identifier | `RB-09` |
| `verification_kind` | `robustness` |
| `evidence_status` | `required` |
| Equations | 1-d gamma-law Euler, primitives \(W=(\rho,u,p)\). Conserved \((\rho,\rho u,E)\). No sources. \(\gamma=1.4\). |
| Oracle | IC only (this increment). Woodward & Colella, *J. Comput. Phys.* 54, 115–173 (1984). Left \((\rho,u,p)=(1,0,1000)\) for \(x<0.1\), middle \((1,0,0.01)\), right \((1,0,100)\) for \(x>0.9\). Usual final time \(t=0.038\) is documented only; this increment does not evolve. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d unit interval \([0,1]\). Physical tube has reflecting walls; the authored in-memory layout uses the periodic helper and is not run. |
| Parameters | \(\gamma=1.4\). \(\rho=1\), \(u=0\). Left \(p=1000\), middle \(p=0.01\), right \(p=100\). Cuts at \(0.1\) and \(0.9\). \(t=0.038\) documented only. Dimensionless. |
| Native dimensions | `POPS_NATIVE_DIM=1`. In-memory path does not load a native artifact. |
| Required capabilities | Cartesian 1-d, uniform, KokkosSerial, MPI off. HLLC/Rusanov + MUSCL when a native series exists. |
| Configurations | Single resolution \(n=32\) for the in-memory report. Authored scheme: Rusanov, MUSCL/VanLeer, SSPRK2, AdaptiveCFL. No reference-fine series. |
| Diagnostics | Three-region IC. Left/right high-pressure locations. Positivity of \(\rho,p\). |
| Thresholds | Empty `orders` with reason containing `interacting blasts / no closed form`. No spatial-order gate. |
| Proves | Documented Woodward–Colella two-blast IC (three regions; left \(p=1000\) at \(x<0.1\); right \(p=100\) at \(x>0.9\); positivity); report renderer accepts empty orders justified by interacting blasts. |
| Does not prove | Observed spatial/temporal order, shock-capturing quality after interaction, reflecting native BCs, AMR, Poisson, coupling, MPI, HLLC vs Rusanov parity, agreement with a uniform-fine or external-code reference. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
