# NO-01 — Native order campaign helper

Infrastructure planner for Serial/OpenMP × Dim1/Dim2 native-order jobs.
In-memory manufactured \(L^2 \propto h^2\) on \(n=16,32,64,128\) only. Live
ROMEO compile/run stays under `verification/machines/`. No `pops.compile`.

| Field | Content |
|---|---|
| Identifier | `NO-01` |
| `verification_kind` | `infrastructure` |
| `evidence_status` | `required` |
| Equations | Manufactured spatial-order stand-in. No conserved system. Live jobs would run TR-01 scalar advection \(\partial_t q + a\partial_x q = 0\). |
| Oracle | Manufactured \(L^2 = h^2\) with \(h=1/n\) on \(n=16,32,64,128\). `exact.py` does not read PoPS output. |
| Domain and boundaries | Unit interval implied by \(h=1/n\). No live mesh. |
| Parameters | Dimensionless. Resolutions \(N=16,32,64,128\). Order gate \(1.8\). |
| Native dimensions | Planner labels `Dim1` and `Dim2`. One live artifact is still exactly one `POPS_NATIVE_DIM`. |
| Required capabilities | None on the in-memory path. Live series needs KokkosSerial + KokkosOpenMP native artefacts on ROMEO. |
| Configurations | `plan_jobs()` emits Serial/OpenMP × Dim1/Dim2. No CFL, flux, reconstruction, or AMR on this path. |
| Diagnostics | Task 15 `observed_order` on the manufactured \(L^2\) series. |
| Thresholds | Observed order \(\ge 1.8\). Manufactured series is exactly \(2\). |
| Proves | Job labels cover Serial/OpenMP × Dim1/Dim2. Manufactured second-order \(L^2\) passes the \(1.8\) gate. Report renderer accepts a NO-01 summary with those spaces and dimensions. |
| Does not prove | Live native spatial order, Dim2 compile, OpenMP thread invariance, MPI, AMR, Poisson, coupling, performance. |
| Resources | Local in-memory helper. Live ROMEO jobs are orchestrator-owned under `verification/machines/`. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record on the in-memory path. |
