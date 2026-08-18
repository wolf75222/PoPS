# TR-04 — face/edge/corner crossing

Phase 3 multiblock transport case. Reuses the TR-02 Gaussian pulse on three
1-d placements of a two-block join whose face is at \(x=0.5\). Exact
translation by whole periods makes the three fields identical. In-memory
only; no live runtime, compile, or ROMEO.

| Field | Content |
|---|---|
| Identifier | `TR-04` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | \(\partial_t q + a\partial_x q = 0\). Conservative scalar \(q\). Canonical \(a=1\). Same IC as TR-02: \(q(x,0)=q_0+A\exp(-\|x-x_0\|^2/(2\sigma^2))\). |
| Oracle | Reused TR-02 manufactured translation via `load_sibling_module` on `verification/cases/transport/gaussian_pulse/exact.py`. Three 1-d placements `face`, `edge`, `corner` sit on the two-block join at \(x=0.5\), one period apart. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Two blocks \([0,0.5]\cup[0.5,1]\). Face at \(0.5\). |
| Parameters | Dimensionless. Default in-memory grid \(n=32\) (16 cells per block). TR-02 defaults \(q_0=0\), \(A=1\), \(x_0=0.37\), \(\sigma=0.08\). Placement times \(t=(0.5-x_0)/a + k\), \(k=0,1,2\). |
| Native dimensions | `POPS_NATIVE_DIM=1` listed for the planner. This increment does not load a native artifact. |
| Required capabilities | None on the in-memory path. A later native series needs a two-block periodic scalar advection Case. |
| Configurations | Face / edge / corner 1-d reductions of the same two-block join. No AMR, no order series in this increment. |
| Diagnostics | Field-to-field \(L^\infty\) between placements. Task 2 volume-weighted L1/L2/L∞ on exact vs exact. |
| Thresholds | Field-to-field \(L^\infty=0\) for exact translation. In-memory exact vs exact \(L^\infty=0\). |
| Proves | TR-02 oracle reuse via `load_sibling_module`. Three 1-d placements share the face at 0.5 and a two-block join. Exact translation maps the placements onto each other. Report renderer accepts a TR-04 summary. |
| Does not prove | Native multiblock flux at the join, 2-d/3-d edge/corner topology, spatial order, AMR, Poisson, coupling, MPI, performance. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
