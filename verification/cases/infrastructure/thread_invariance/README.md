# IF-02 — OpenMP thread-count invariance

Phase 0 infrastructure contract. Reuses the TR-01 manufactured sine and
samples it independently on OpenMP-style static slices as if run with 1, 2,
4, and 8 threads. The arrays are identical. Field-to-field \(L^\infty\) is 0.
In-memory only; no live OpenMP, compile, or Kokkos thread sweep. A live
Kokkos OpenMP thread sweep is ROMEO-only.

| Field | Content |
|---|---|
| Identifier | `IF-02` |
| `verification_kind` | `infrastructure` |
| `evidence_status` | `required` |
| Equations | \(\partial_t q + a\partial_x q = 0\). Conservative scalar \(q\). Canonical \(a=1\). Same IC as TR-01: \(q=q_0+\varepsilon\sin(2\pi k x)\). |
| Oracle | Reused TR-01 manufactured sine via `load_sibling_module` on `verification/cases/transport/advection_sine/exact.py`. Sampled independently on each static thread slice. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Thread labels \(1,2,4,8\) partition the same \(N\) cells by static OpenMP-style slices. |
| Parameters | Dimensionless. Default in-memory grid \(N=32\) (divisible by every thread count). TR-01 defaults \(q_0=1\), \(\varepsilon=10^{-2}\), \(k=1\), \(a=1\). |
| Native dimensions | `POPS_NATIVE_DIM=1` listed for the planner. This increment does not load a native artifact. |
| Required capabilities | None on the in-memory path. Optional `run_native_threads((1, 2, 4, 8), n_cells=32, t_end=0.25)` sets `OMP_NUM_THREADS` and reuses TR-01 `run_native` via `load_sibling_module`. A live Kokkos OpenMP thread sweep on ROMEO remains the machine campaign. |
| Configurations | Four thread labels. Physics, resolution, and \(t\) stay fixed. Only the static slice count changes. |
| Diagnostics | Pairwise field-to-field L1/L2/L∞ between the four exact thread labels. |
| Thresholds | Difference between thread labels is \(L^\infty=0\). No spatial-order gate on the in-memory path. |
| Proves | Exact fields are independent of the OpenMP thread-count label. Report renderer accepts an IF-02 summary with empty `orders` and reason `exact-field identity / no live OpenMP`. |
| Does not prove | Live OpenMP / Kokkos thread-count invariance of a compiled Program (ROMEO-only; optional local wrapper is not the campaign), MPI/GPU invariance, spatial order, AMR, Poisson, coupling, performance. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
