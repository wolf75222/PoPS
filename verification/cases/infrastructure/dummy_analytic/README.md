# PH-00 — Phase 0 dummy analytic cosine

Infrastructure dummy. In-memory manufactured cosine only. No solver, no compile,
no bind, no ROMEO.

| Field | Content |
|---|---|
| Identifier | `PH-00` |
| `verification_kind` | `infrastructure` |
| `evidence_status` | `required` |
| Equations | Manufactured scalar field \(u(x)=\cos(2\pi x)\) on a uniform 1-d cell grid. No conserved system, no sources. |
| Oracle | `manufactured_cosine`: cell-center samples of \(u(x)\) on \([0,1]\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Dummy samples the interior cells only. |
| Parameters | \(n=32\) uniform cells. Dimensionless. Optional numerical perturbation \(10^{-12}\). |
| Native dimensions | `POPS_NATIVE_DIM=1` only. This dummy does not load a native artifact. |
| Required capabilities | None (`requires = []`). KokkosSerial listed for the planner; MPI off. |
| Configurations | Single resolution \(n=32\). No blocks, CFL, integrator, flux, reconstruction, or AMR. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ of the manufactured pair (volumes = cell widths). |
| Thresholds | `finite = true`. No spatial-order gate. |
| Proves | Phase 0 report renderer can emit a complete schema-valid campaign report. Planner still refuses more than two nodes. |
| Does not prove | Spatial order, AMR, Poisson, coupling, MPI, performance. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
