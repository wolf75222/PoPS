# IF-03 — Kokkos Serial vs OpenMP parity

Phase 0 infrastructure contract plus an optional Phase 6 native compare.
Reuses the TR-01 manufactured sine and samples the same exact field once
under the `KokkosSerial` label and once under the `KokkosOpenMP` label.
Field-to-field \(L^\infty\) is 0. In-memory tests stay exact-field identity.

Optional `run_native` / `run_native_spaces` reuse TR-01 `run_native` at
`OMP_NUM_THREADS=1` (Serial-like) and `OMP_NUM_THREADS=8` (OpenMP), matching
`verification/machines/run_native_order.py`. `KokkosCuda` / `KokkosHIP` raise
`NativeUnavailable` (`no public CUDA space`) before a native run.

| Field | Content |
|---|---|
| Identifier | `IF-03` |
| `verification_kind` | `infrastructure` |
| `evidence_status` | `required` |
| Equations | \(\partial_t q + a\partial_x q = 0\). Conservative scalar \(q\). Canonical \(a=1\). Same IC as TR-01: \(q=q_0+\varepsilon\sin(2\pi k x)\). |
| Oracle | Reused TR-01 manufactured sine via `load_sibling_module` on `verification/cases/transport/advection_sine/exact.py`. Sampled independently under each Kokkos execution-space label. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Labels: `KokkosSerial` and `KokkosOpenMP`. Physics, resolution, and \(t\) stay fixed. Only the execution-space name changes. |
| Parameters | Dimensionless. Default in-memory grid \(N=32\). TR-01 defaults \(q_0=1\), \(\varepsilon=10^{-2}\), \(k=1\), \(a=1\). |
| Native dimensions | `POPS_NATIVE_DIM=1`. Optional native compare reuses TR-01 `run_native` and refuses any other artifact dim. |
| Required capabilities | None on the in-memory path. Live Serial/OpenMP compare needs a Dim1 artifact + compiler/Kokkos. GPU is capability-gated (`no public CUDA space`). |
| Configurations | Two execution-space labels. Physics, resolution, and \(t\) stay fixed. `case.toml` `execution_spaces = ["KokkosSerial", "KokkosOpenMP"]`. |
| Diagnostics | Field-to-field L1/L2/L∞ between the two labelled exact fields. |
| Thresholds | Difference between labels is \(L^\infty=0\). No spatial-order gate on the in-memory path. |
| Proves | Exact fields are independent of the Kokkos Serial vs OpenMP label. Optional native path compares the same TR-01 field at 1 vs 8 OpenMP threads. Report renderer accepts an IF-03 summary with empty `orders` and reason `exact-field identity / no live Kokkos`. |
| Does not prove | Live CUDA/HIP parity (refused), MPI invariance, spatial order, AMR, Poisson, coupling, performance. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
