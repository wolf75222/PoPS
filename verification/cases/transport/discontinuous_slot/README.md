# TR-07 — discontinuous slot

Phase 1 discontinuous scalar advection. Exact solution is a periodic
translation of a rectangular slot. In-memory only; no live runtime, compile,
or ROMEO. The 2-d slotted disk remains later.

| Field | Content |
|---|---|
| Identifier | `TR-07` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | \(\partial_t q + a\partial_x q = 0\). Conservative scalar \(q\). No sources. Canonical \(a=1\). |
| Oracle | 1-d periodic slot \(q=1\) for \(\lvert x-x_0\rvert < w/2\), else \(0\). Canonical \(x_0=0.5\), \(w=0.25\). Exact translation \(q(x,t)=q_0(x-a t)\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). |
| Parameters | Dimensionless. Default in-memory grid \(N=64\) (slot edges sit on cell faces). |
| Native dimensions | `POPS_NATIVE_DIM=1` listed for the planner. This increment does not load a native artifact. |
| Required capabilities | None on the in-memory path. A later native series needs a limiter on a discontinuous IC. |
| Configurations | Single 1-d resolution. Manufactured smeared-vs-exact pair plants overshoot \(1.1\) and a negative undershoot. |
| Diagnostics | Periodic total variation. Overshoot \(\max(0,\max q-\max q_{\mathrm{ex}})\). Undershoot \(\max(0,\min q_{\mathrm{ex}}-\min q)\). |
| Thresholds | Exact-slot TV \(=2\). A field with a \(1.1\) spike reports overshoot \(>0\). Empty `orders` with reason `discontinuous / limiter, not order-2`. |
| Proves | Periodic translation identity of the slot. TV of the exact slot is 2. Overshoot helper flags a \(1.1\) spike. Report renderer accepts a TR-07 summary with empty orders. |
| Does not prove | Native limiter quality, observed spatial order, 2-d slotted disk, AMR, Poisson, coupling, MPI, performance. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
