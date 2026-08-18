# IF-08 — exact native-dim specialization

Infrastructure planner guard. A requested dimension that does not match
`POPS_NATIVE_DIM` is refused before any fake run. Matching `dim=1` emits
TR-01. Reuses `expand_jobs` / `resolve_artifact_dim`.

Optional Phase 6 helpers: `run_doctor()` wraps `pops.doctor()`. Presenting
the Dim2 case (GE-03) under `POPS_NATIVE_DIM=1` raises `NativeUnavailable`
before a fake run or GE-03 `run_native`.

| Field | Content |
|---|---|
| Identifier | `IF-08` |
| `verification_kind` | `infrastructure` |
| `evidence_status` | `required` |
| Equations | Planner-only. The emitted scientific case is TR-01 (\(\partial_t q + a\partial_x q = 0\)), but this increment does not advance a field. |
| Oracle | Task 10 campaign helpers. `expand_jobs` with `artifact_dim=1` and requested `[2]` raises `CampaignError`. Matching `dim=1` emits `CampaignJob(TR-01, 1)`. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d planner record inherited from TR-01 (`native_dimensions = [1]`). No mesh is built. |
| Parameters | Artifact dim `1`. Requested mismatch `[2]`. Matching request `[1]`. Dimensionless. |
| Native dimensions | `POPS_NATIVE_DIM=1` listed for the planner. A Dim2 case under that artifact raises `NativeUnavailable` before a fake/native run. There is no fallback to another extension. |
| Required capabilities | None (`requires = []`). KokkosSerial listed for the planner; MPI off. |
| Configurations | Single planner case TR-01. No solver, CFL, integrator, flux, reconstruction, or AMR. |
| Diagnostics | Raise vs emit. Fake-run counter stays 0 on mismatch. |
| Thresholds | Mismatch refused (message names `POPS_NATIVE_DIM` / no fallback). Matching emit is TR-01 at dim 1. Empty `orders` with reason `exact native-dim specialization / no live native artifact`. |
| Proves | A requested dim that differs from the loaded artifact is refused before a fake run. Matching dim 1 plans TR-01. A Dim2 case under `POPS_NATIVE_DIM=1` raises `NativeUnavailable` before a fake/native run. Report renderer accepts an IF-08 summary. |
| Does not prove | Live native compile, `select_native_dimension` success on a Dim2 artifact, spatial/temporal order, MPI, AMR, Poisson, coupling, performance. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
