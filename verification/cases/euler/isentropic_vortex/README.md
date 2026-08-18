# EU-02 — Isentropic vortex (2-d)

Classic isentropic vortex on a periodic box. The exact solution at time \(t\)
is the initial vortex translated by the free-stream velocity \((u_\infty,v_\infty)\).
1-d is not applicable. Native compile is optional.

| Field | Content |
|---|---|
| Identifier | `EU-02` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | 2-d gamma-law Euler, primitives \(W=(\rho,u,v,p)\). Conserved \((\rho,\rho u,\rho v,E)\). No sources. \(\gamma=1.4\). |
| Oracle | Yee/Sandham/Djomehri isentropic vortex: background \((\rho_\infty,u_\infty,v_\infty,p_\infty)\) plus isentropic perturbation. Exact at \(t\) is translation by \((u_\infty,v_\infty)\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 2-d periodic box \([0,10]^2\). 1-d is not applicable. |
| Parameters | \(\rho_\infty=1\), \(p_\infty=1\), \((u_\infty,v_\infty)=(1,0)\) canonical, \(\beta=5\), centre \((5,5)\). Dimensionless. |
| Native dimensions | `POPS_NATIVE_DIM=2`. In-memory path does not load a native artifact. |
| Required capabilities | Cartesian 2-d, uniform, periodic, KokkosSerial, MPI off. HLLC/Rusanov + MUSCL when a native series exists. |
| Configurations | Single resolution \(n=32^2\) for the in-memory report. Authored scheme: Rusanov, MUSCL/VanLeer, SSPRK2, AdaptiveCFL. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ on density of exact vs exact. Density/pressure positivity. Translation by \((1,0)\). Entropy function \(p/\rho^\gamma\) constant. |
| Thresholds | Exact-vs-exact L∞ = 0. No spatial-order gate on the in-memory path. |
| Proves | Isentropic-vortex oracle (positivity; exact translation by \((u_\infty,v_\infty)\); \(p/\rho^\gamma\) invariant); report renderer. |
| Does not prove | Observed spatial/temporal order, AMR, Poisson, coupling, MPI, HLLC vs Rusanov parity, 1-d reduction. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
