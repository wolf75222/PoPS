# AM-04 — AMR subcycling

Phase 3 AMR clock contract. Documents that the fine-level step is the
coarse step divided by the declared ratio. Manufactured temporal error
is \(E\propto\Delta t_{\mathrm{fine}}^2\). In-memory helpers remain; a
public 1-d periodic AMR Case validates and resolves for ratios 1 and 2
(ratio 4 when the public API accepts it). `run_native` compiles, binds,
and runs when Kokkos and a compiler are present.

| Field | Content |
|---|---|
| Identifier | `AM-04` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | Two-level AMR clocks. Coarse step \(\Delta t_{\mathrm{coarse}}\). Fine step \(\Delta t_{\mathrm{fine}}=\Delta t_{\mathrm{coarse}}/r\). Temporal error manufactured as RK2, \(E\propto\Delta t_{\mathrm{fine}}^2\). Live physics is TR-01 scalar sine advection \(a=1\). |
| Oracle | Ratios \(r\in\{1,2,4\}\). `fine_steps_per_coarse(r)=r`, so ratio 2 has 2 fine steps per coarse. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\) via `CartesianDomain` + `Cartesian1D`. Static two-level tag on \(x>0.5\). Frozen regrid. Clock contract stays available without a mesh. |
| Parameters | Dimensionless. Default \(\Delta t_{\mathrm{coarse}}=1/128\). Ratios 1, 2, 4. Default \(n_{\mathrm{coarse}}=8\). |
| Native dimensions | `POPS_NATIVE_DIM=1`. `run_native` returns the level-0 tracer as shape `(n_coarse,)`. |
| Required capabilities | Public AMR (`layouts.AMR`, `AMRHierarchy(max_levels=2, ratios=(2,))`), `AMRExecution.synchronous` (ratio 1) or `AMRExecution.subcycled` with `AMRClockRelation(0, 1, r)` (ratios 2 and 4), Cartesian 1-d periodic, SSPRK2. |
| Configurations | Ratio set 1, 2, 4. Frozen spatial marker on the right half. No moving-patch or reflux claim. |
| Diagnostics | Fine-step count per coarse step. Task 15 observed order on the manufactured \(\Delta t_{\mathrm{fine}}\) series (`kind=temporal`). |
| Thresholds | Ratio 2 → 2 fine steps/coarse. Observed order \(\ge 1.8\) for declared RK2 (plan §8.2). Manufactured \(E\propto\Delta t_{\mathrm{fine}}^2\) yields order 2. |
| Proves | Fine dt = coarse_dt / ratio for 1, 2, 4. Ratio 2 has 2 fine steps per coarse. Manufactured temporal order 2 vs \(\Delta t_{\mathrm{fine}}\). Public 1-d periodic AMR Case validates and resolves for ratios 1 and 2. Report renderer accepts an AM-04 summary. |
| Does not prove | Spatial order retention, reflux, interface error, MPI, performance. Native compile requires Kokkos + a compiler. Ratio 4 resolve is attempted and documented if refused. |
| Resources | Local authoring and in-memory clock contract. Native path is optional. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compile records compiler/Kokkos only when `run_native` runs. |
