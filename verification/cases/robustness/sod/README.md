# RB-01 — Sod shock tube

Classic 1-d Sod Riemann problem. Oracle is the exact gamma-law Riemann
solution (Toro Ch. 4). `run_native` compiles, binds, and advances the
authored Case when Kokkos and a compiler exist.

| Field | Content |
|---|---|
| Identifier | `RB-01` |
| `verification_kind` | `robustness` |
| `evidence_status` | `required` |
| Equations | 1-d gamma-law Euler, primitives \(W=(\rho,u,p)\). Conserved \((\rho,\rho u,E)\). No sources. \(\gamma=1.4\). |
| Oracle | Exact Riemann problem. Left \((\rho,u,p)=(1,0,1)\), right \((0.125,0,0.1)\), diaphragm \(x_0=0.5\). Star states from \(f_L(p^*)+f_R(p^*)+(u_R-u_L)=0\) (Toro 4.5); Sod is left rarefaction + contact + right shock. Positions at \(t=0.2\): \(x_{\mathrm{head}}=x_0+(u_L-c_L)t\), \(x_{\mathrm{tail}}=x_0+(u^*-c_L^*)t\), \(x_{\mathrm{contact}}=x_0+u^*t\), \(x_{\mathrm{shock}}=x_0+S_Rt\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d unit interval \([0,1]\). Physical tube is transmissive. No public Euler transmissive/outflow BC is resolved here, so `resolve_plan` / `run_native` keep `uniform_periodic_layout`. |
| Parameters | \(\gamma=1.4\). \(t=0.2\). Dimensionless. |
| Native dimensions | `POPS_NATIVE_DIM=1`. `run_native` returns conserved \((\rho,\rho u,E)\) as shape `(3, n)`. |
| Required capabilities | Cartesian 1-d, uniform, KokkosSerial, MPI off. HLLC/Rusanov + MUSCL when a native series exists. |
| Configurations | Single resolution \(n=32\) for the in-memory report. Authored scheme: Rusanov, MUSCL/VanLeer, SSPRK2, AdaptiveCFL. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ on density of exact vs exact. Positivity of \(\rho,p\). Left/right ICs. |
| Thresholds | Exact-vs-exact L∞ = 0. Empty `orders` with reason containing `shock`. No spatial-order gate. |
| Proves | Exact Riemann oracle (positivity; left/right ICs); report renderer accepts empty orders justified by a shock. |
| Does not prove | Observed spatial/temporal order, shock-capturing quality, AMR, Poisson, coupling, MPI, HLLC vs Rusanov parity, transmissive native BCs. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compile records compiler/Kokkos only when `run_native` runs. |
