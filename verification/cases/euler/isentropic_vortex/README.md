# EU-02 — Isentropic vortex (2-d)

Classic Yee/Sandham/Djomehri isentropic vortex on a periodic box. The exact
solution at time \(t\) is the initial vortex translated by
\((u_\infty,v_\infty)\). 1-d is not applicable.

| Field | Content |
|---|---|
| Identifier | `EU-02` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | 2-d gamma-law Euler. Primitives \(W=(\rho,u,v,p)\). Conserved \(U=(\rho,\rho u,\rho v,E)\). No sources. \(\gamma=1.4\). Sign: \(E=p/(\gamma-1)+\tfrac12\rho(u^2+v^2)\). Vorticity \(\omega=\partial_x v-\partial_y u\). |
| Oracle | Independent `exact.py`. Primitive and conserved oracles are analytic Gauss–Legendre cell averages of the translated vortex. Point samples are not cell averages. `exact.py` does not read PoPS output. |
| Domain and boundaries | Periodic box \([0,10]^2\). 1-d is not applicable. 3-d is extended, not claimed here. |
| Parameters | \(\rho_\infty=1\), \(p_\infty=1\), canonical \((u_\infty,v_\infty)=(1,0)\), \(\beta=5\), centre \((5,5)\). Dimensionless. |
| Native dimensions | `POPS_NATIVE_DIM=2`. |
| Required capabilities | Cartesian 2-d, uniform, periodic, KokkosSerial, MPI off. Acceptance scheme: Rusanov + **WENO5-Z** + SSPRK2. VanLeer is a labeled TVD fail control only. |
| Configurations | `canonical` global: AdaptiveCFL \(C=0.4\), \(t=1\), \(n=16,32,64,128,256\). Isolated spatial: FixedDt \(\Delta t=0.16 h^2\) on the same meshes; provenance CFL is \(\Delta t/h\) per mesh. Temporal: FixedDt after WENO, only when spatial \(L^\infty\) is \(\ge 10\times\) below the coarsest-dt error (honest isolation; not claimed if that fails). OpenMP/MPI: n=16 smokes only. |
| Diagnostics | L1/L2/L∞ on \(\rho,u,v,p\) vs primitive cell averages; conserved norms vs conserved cell averages; vortex-centre / phase; vorticity max; radial_anisotropy and xy_symmetry separately; mass / momentum / energy conservation. |
| Thresholds | §9.3: keep every interval as evidence (including the pre-asymptotic 32→64 dip). Gate the last two usable L∞ intervals (64→128 and 128→256 on the five-mesh series) at \(\ge 1.8\). Threshold is never lowered. Finite-only is not acceptance. |
| Proves | Native 2-d periodic Euler advection of a smooth isentropic vortex against cell-average oracles; conservation residuals of the public pipeline; centre tracking. |
| Does not prove | AMR (static or dynamic), 3-d, HLLC vs Rusanov, isolated spatial unless `family=spatial`, temporal order unless the series is isolated, MPI/OpenMP invariance from a smoke. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. ROMEO ≤ 2 nodes. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Outputs under `build/verification/`. |
| Typical failures | Point-sample IC vs cell-average oracle (false first-order); constant-CFL labelled spatial; exact-vs-exact L∞=0; injected orders. |
