# RB-08 — Double Mach reflection

Woodward–Colella Mach 10 double Mach reflection. Geometry helpers only
(pre/post Rankine–Hugoniot states and the \(t=0\) shock-front line). Native
compile is optional.

| Field | Content |
|---|---|
| Identifier | `RB-08` |
| `verification_kind` | `robustness` |
| `evidence_status` | `required` |
| Equations | 2-d gamma-law Euler, primitives \(W=(\rho,u,v,p)\). Conserved \((\rho,\rho u,\rho v,E)\). No sources. \(\gamma=1.4\). |
| Oracle | Geometry helpers only. No late-time closed form. Pre-shock \((\rho,u,p)=(1.4,0,1)\). Post-shock from the normal Rankine–Hugoniot jump for \(M=10\). Shock/wedge angle \(30^\circ\). In the Woodward–Colella box the reflecting wall is the \(x\)-axis and the shock-wall incidence is \(60^\circ\); the \(t=0\) front is \(x(y)=1/6+y/\tan 60^\circ\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 2-d box \([0,4]\times[0,1]\). Physical DMR needs a reflecting wall from \(x=1/6\) and a time-dependent top; the authored in-memory layout uses the periodic helper and is not run. |
| Parameters | \(\gamma=1.4\). \(M=10\). Wedge / shock angle \(30^\circ\). \(t=0.2\). Dimensionless. |
| Native dimensions | `POPS_NATIVE_DIM=2`. In-memory path does not load a native artifact. |
| Required capabilities | Cartesian 2-d, uniform, KokkosSerial, MPI off. HLLC/Rusanov + MUSCL when a native series exists. |
| Configurations | Single resolution \(n=32^2\) for the in-memory report. Authored scheme: Rusanov, MUSCL/VanLeer, SSPRK2, AdaptiveCFL. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ on density of IC vs IC. Rankine–Hugoniot post-shock \(\rho,p,u\). Shock angle \(30^\circ\). Positivity of \(\rho,p\). |
| Thresholds | IC-vs-IC L∞ = 0. Empty `orders` with reason containing `DMR morphology / no closed form`. No spatial-order gate. |
| Proves | Documented DMR IC (RH post-shock for \(M=10\) into \((1.4,0,1)\); \(30^\circ\) shock/wedge; positivity); report renderer accepts empty orders justified by DMR morphology. |
| Does not prove | Observed spatial/temporal order, late-time DMR morphology, shock-capturing quality, AMR, Poisson, coupling, MPI, reflecting/time-dependent native BCs. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
