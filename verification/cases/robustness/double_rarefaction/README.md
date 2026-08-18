# RB-02 — Double rarefaction / near-vacuum

1-d Toro Test 2 (123 problem). Two rarefactions leave a low-density star.
Oracle is the exact gamma-law Riemann solution (Toro Ch. 4). Native compile
is optional.

| Field | Content |
|---|---|
| Identifier | `RB-02` |
| `verification_kind` | `robustness` |
| `evidence_status` | `required` |
| Equations | 1-d gamma-law Euler, primitives \(W=(\rho,u,p)\). Conserved \((\rho,\rho u,E)\). No sources. \(\gamma=1.4\). |
| Oracle | Exact Riemann problem. Left \((\rho,u,p)=(1,-2,0.4)\), right \((1,2,0.4)\), diaphragm \(x_0=0.5\). Star states from \(f_L(p^*)+f_R(p^*)+(u_R-u_L)=0\) (Toro 4.5); both waves are rarefactions into a low-density star. Positions at \(t=0.15\): \(x_{\mathrm{head},L}=x_0+(u_L-c_L)t\), \(x_{\mathrm{tail},L}=x_0+(u^*-c_L^*)t\), \(x_{\mathrm{contact}}=x_0+u^*t\), \(x_{\mathrm{tail},R}=x_0+(u^*+c_R^*)t\), \(x_{\mathrm{head},R}=x_0+(u_R+c_R)t\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d unit interval \([0,1]\). Physical tube is transmissive; the authored in-memory layout uses the periodic helper and is not run. |
| Parameters | \(\gamma=1.4\). \(t=0.15\). Dimensionless. |
| Native dimensions | `POPS_NATIVE_DIM=1`. In-memory path does not load a native artifact. |
| Required capabilities | Cartesian 1-d, uniform, KokkosSerial, MPI off. Rusanov + FirstOrder when a native series exists. |
| Configurations | Single resolution \(n=32\) for the in-memory report. Authored scheme: Rusanov, FirstOrder, SSPRK2, AdaptiveCFL, `FiniteVolume(positivity_floor=1e-8)`. MUSCL/VanLeer at the \(u=\pm 2\) diaphragm reconstructs \(p<0\) and publishes `InvalidWaveSpeed` (status=6) on the first residual. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ on density of exact vs exact. Positivity of \(\rho,p\) and of \(p^*,\rho^*\). Star density below both sides. |
| Thresholds | Exact-vs-exact L∞ = 0. Empty `orders` with reason containing `shock/rarefaction`. No spatial-order gate. |
| Proves | Exact Riemann oracle (positivity; low-density star below both sides); report renderer accepts empty orders justified by a shock/rarefaction. |
| Does not prove | Observed spatial/temporal order, near-vacuum robustness of a discrete solver, AMR, Poisson, coupling, MPI, HLLC vs Rusanov parity, transmissive native BCs. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
