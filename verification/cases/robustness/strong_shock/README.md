# RB-03 — Strong shock

Toro/Castro 1-d strong-shock Riemann problem. Oracle is the exact
gamma-law Riemann solution (Toro Ch. 4; same star-state machinery as
RB-01). Native compile is optional.

| Field | Content |
|---|---|
| Identifier | `RB-03` |
| `verification_kind` | `robustness` |
| `evidence_status` | `required` |
| Equations | 1-d gamma-law Euler, primitives \(W=(\rho,u,p)\). Conserved \((\rho,\rho u,E)\). No sources. \(\gamma=1.4\). |
| Oracle | Exact Riemann problem. Left \((\rho,u,p)=(1,0,1000)\), right \((1,0,0.01)\), diaphragm \(x_0=0.5\). Star states from \(f_L(p^*)+f_R(p^*)+(u_R-u_L)=0\) (Toro 4.5); left rarefaction + contact + strong right-going shock (\(p^*\gg p_R\)). Positions at \(t=0.012\): \(x_{\mathrm{head}}=x_0+(u_L-c_L)t\), \(x_{\mathrm{tail}}=x_0+(u^*-c_L^*)t\), \(x_{\mathrm{contact}}=x_0+u^*t\), \(x_{\mathrm{shock}}=x_0+S_Rt\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d unit interval \([0,1]\). Physical tube is transmissive; the authored in-memory layout uses the periodic helper and is not run. |
| Parameters | \(\gamma=1.4\). \(t=0.012\). Dimensionless. |
| Native dimensions | `POPS_NATIVE_DIM=1`. In-memory path does not load a native artifact. |
| Required capabilities | Cartesian 1-d, uniform, KokkosSerial, MPI off. HLLC/Rusanov + MUSCL when a native series exists. |
| Configurations | Single resolution \(n=32\) for the in-memory report. Authored scheme: Rusanov, MUSCL/VanLeer, SSPRK2, AdaptiveCFL. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ on density of exact vs exact. Positivity of \(\rho,p\). \(p^*\gg p_R\). Right-going shock. |
| Thresholds | Exact-vs-exact L∞ = 0. \(p^*/p_R>1000\). \(S_R>0\). Empty `orders` with reason containing `shock`. No spatial-order gate. |
| Proves | Exact Riemann oracle (positivity; strong right shock \(p^*\gg p_R\)); report renderer accepts empty orders justified by a shock. |
| Does not prove | Observed spatial/temporal order, shock-capturing quality, AMR, Poisson, coupling, MPI, HLLC vs Rusanov parity, transmissive native BCs. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
