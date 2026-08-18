# PF-06 — Euler–Poisson step stand-in

Phase 0 pipeline-segment stand-in plus optional CP-02 native timing. A toy
Euler–Poisson step is split into halo, hyperbolic, charge, poisson,
gradient, and source. Each stage records a fake timing \(>0\). The seventh
segment is `total` and equals the sum. Optional ``run_native`` times CP-02
``verification/cases/euler_poisson/langmuir_cold/run.py`` and returns
elapsed plus cells/s. GPU spaces are refused.

| Field | Content |
|---|---|
| Identifier | `PF-06` |
| `verification_kind` | `infrastructure` |
| `evidence_status` | `required` |
| Equations | No PDE is advanced. The stand-in only partitions a fake EP-step wall time. |
| Oracle | Seven named segments. Pipeline stages `halo`, `hyperbolic`, `charge`, `poisson`, `gradient`, `source` each have a deterministic fake timing \(>0\). `total` is their exact sum. `exact.py` does not read PoPS output. |
| Domain and boundaries | None. No mesh is allocated. |
| Parameters | Dimensionless fake seconds. Six pipeline stages plus `total`. |
| Native dimensions | `POPS_NATIVE_DIM=1` listed for the planner. Optional `run_native` reuses CP-02 Dim1. |
| Required capabilities | None on the in-memory path. Native timing needs CP-02 compile (Kokkos). No public CUDA space. |
| Configurations | Single toy pipeline. Optional CP-02 timing. No AMR, no GPU, no two-node. |
| Diagnostics | Segment count. Per-stage fake times. `total` versus the pipeline sum. Optional CP-02 `elapsed_s` / `cells_per_second`. |
| Thresholds | Seven segments present. Every pipeline stage \(>0\). `total` equals the sum exactly. |
| Proves | The toy EP step exposes seven segments and a partition of the fake total. Report renderer accepts a PF-06 summary with empty `orders`. `run_native` times CP-02 or refuses GPU / missing Kokkos. |
| Does not prove | Per-segment native timers, live halo/Poisson/source kernels, AMR, MPI, GPU, or a two-node cells/s claim. Native elapsed includes CP-02 compile+bind+run. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
