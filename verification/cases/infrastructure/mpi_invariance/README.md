# IF-01 — MPI decomposition invariance

Phase 0 infrastructure contract. Reuses the TR-01 manufactured sine and
samples it independently on one block \([0,1]\), two blocks
\([0,0.5]\cup[0.5,1]\), and the 1×4 / 4×1 1-d splits at \(0.25/0.5/0.75\).
Field-to-field \(L^\infty\) is 0. In-memory only; no live MPI, compile, or
ROMEO.

| Field | Content |
|---|---|
| Identifier | `IF-01` |
| `verification_kind` | `infrastructure` |
| `evidence_status` | `required` |
| Equations | \(\partial_t q + a\partial_x q = 0\). Conservative scalar \(q\). Canonical \(a=1\). Same IC as TR-01: \(q=q_0+\varepsilon\sin(2\pi k x)\). |
| Oracle | Reused TR-01 manufactured sine via `load_sibling_module` on `verification/cases/transport/advection_sine/exact.py`. Sampled independently on each block of a 1-d MPI-style split. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Placements: one block \([0,1]\); two blocks \([0,0.5]\cup[0.5,1]\); 1×4 and 4×1 four-block splits at \(x=0.25,0.5,0.75\). |
| Parameters | Dimensionless. Default in-memory grid \(N=32\) (all joins are cell faces). TR-01 defaults \(q_0=1\), \(\varepsilon=10^{-2}\), \(k=1\), \(a=1\). |
| Native dimensions | `POPS_NATIVE_DIM=1` listed for the planner. This increment does not load a native artifact. |
| Required capabilities | None on the in-memory path. Optional `run_native` raises `NativeUnavailable` (`public multi-rank Uniform/MPI layout is not available`). `pops.layouts` has `Uniform` / `AMR` only; `Uniform` has no rank argument. Do not invent `mpirun`. |
| Configurations | Four 1-d placements. Physics, resolution, and \(t\) stay fixed. Only the block edges move. 1×4 and 4×1 share the same 1-d splits. |
| Diagnostics | Pairwise field-to-field L1/L2/L∞ between the four exact placements. |
| Thresholds | Difference between decompositions is \(L^\infty=0\). No spatial-order gate on the in-memory path. |
| Proves | Exact fields are independent of the 1-d MPI-style split. Report renderer accepts an IF-01 summary with empty `orders` and reason `exact-field identity / no live MPI`. |
| Does not prove | Live MPI communication, rank-count invariance of a compiled Program, thread/GPU invariance, spatial order, AMR, Poisson, coupling, performance. Native comparison is refused until a public multi-rank Uniform/MPI layout exists; TR-01 `run_native` stays single-rank Uniform. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
