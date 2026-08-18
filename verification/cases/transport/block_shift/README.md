# TR-05 — translated block faces

Phase 3 same-level ghost-face contract. The physical problem is the TR-02
Gaussian pulse. Only the two-block join is translated. In-memory only; no
live runtime, compile, or ROMEO.

| Field | Content |
|---|---|
| Identifier | `TR-05` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | \(\partial_t q + a\partial_x q = 0\) with the TR-02 pulse \(q(x,0)=A\exp(-\|x-x_0\|^2/(2\sigma^2))\). Canonical \(a=1\), \(A=1\), \(x_0=0.37\), \(\sigma=0.08\). |
| Oracle | Same manufactured translation as TR-02. Sampled independently on each side of a cell-face join. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Two blocks joined at a cell face. |
| Parameters | Canonical interfaces \(x=0.25\), \(0.2578125\), \(0.375\). Default in-memory grid \(N=128\) (all three joins are faces). Block sizes \(8,16,32,64\) are documented for a later native series. |
| Native dimensions | `POPS_NATIVE_DIM=1` listed for the planner. This increment does not load a native artifact. |
| Required capabilities | None on the in-memory path. A later native series needs same-level ghost fill on translated joins. |
| Configurations | Three 1-d two-block placements. Physics, resolution, and \(\Delta t\) stay fixed. Only the join moves. |
| Diagnostics | Pairwise field-to-field L1/L2/L∞ between the three exact placements. |
| Thresholds | Difference between decompositions is \(L^\infty=0\). No spatial-order gate on the in-memory path. |
| Proves | Exact fields are independent of the translated join. Report renderer accepts a TR-05 summary. |
| Does not prove | Native ghost-fill, MPI communication, incomplete stencils, alignment bugs, AMR, Poisson, coupling, performance. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
