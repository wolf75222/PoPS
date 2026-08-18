# AM-06 — level count

Phase 3 AMR capability contract. Documents that this campaign materializes at
most three hierarchy levels. Request 1, 2, or 3 is accepted. Request 4 is
refused with `requested_level_count` and `supported_level_count` in the
message, before any compile artefact. The in-memory `request_level_count`
gate stays. A public 1-d periodic AMR Case (copied from AM-01) validates and
resolves for 1–3 levels. `run_native` compiles, binds, and runs when Kokkos
and a compiler are present.

| Field | Content |
|---|---|
| Identifier | `AM-06` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | AMR hierarchy authoring. The campaign asks for a derived level count; the provider advertises a maximum materialized level count. Live physics is TR-01 scalar sine advection \(a=1\). |
| Oracle | Documented `supported_levels = 3`. Request \(n\in\{1,2,3\}\) returns \(n\). Request 4 raises with `requested_level_count=4` and `supported_level_count=3` in the message. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\) via `CartesianDomain` + `Cartesian1D`. Static tag on \(x>0.5\). Frozen regrid. The in-memory gate stays hierarchy-local. |
| Parameters | Dimensionless. `SUPPORTED_LEVELS = 3`. Requests 1, 2, 3, 4. Default \(n_{\mathrm{coarse}}=8\). |
| Native dimensions | `POPS_NATIVE_DIM=1`. `run_native` returns the level-0 tracer as shape `(n_coarse,)`. |
| Required capabilities | Level 1 is Uniform (`layouts.Uniform`). Levels 2–3 are public AMR (`layouts.AMR`): `AMRHierarchy(max_levels=2, ratios=(2,))` and `AMRHierarchy(max_levels=3, ratios=(2,2))`. Cartesian 1-d periodic, SSPRK2. |
| Configurations | Level-count gate plus public resolve. Frozen spatial marker on the right half. No interface placement, subcycling-order, or reflux claim. |
| Diagnostics | Accepted level count. Refusal message contains both count tokens. |
| Thresholds | Request 1, 2, 3 → accepted. Request 4 → refused before artefact. Do not silently drop 4 to 3. |
| Proves | Documented support of three levels. Request 4 is a detectable over-request. Public 1-d periodic AMR Case validates and resolves for 1, 2, and 3 levels. Report renderer accepts an AM-06 summary. |
| Does not prove | Native hierarchy materialization, AM-01 / AM-02 / AM-04 numerics, MPI, performance. Native compile requires Kokkos + a compiler. |
| Resources | Local authoring and in-memory contract. Native path is optional. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compile records compiler/Kokkos only when `run_native` runs. |
