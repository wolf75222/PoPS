# PF-09 — Load-balance stand-in

Phase 1 performance stand-in. Distribute 64 cells over 4 ranks: uniform
occupancy versus one hotspot of 40 cells on rank 0. Report max, mean, and
population coefficient of variation. In-memory only; no live runtime,
compile, MPI, or timed PF run.

| Field | Content |
|---|---|
| Identifier | `PF-09` |
| `verification_kind` | `infrastructure` |
| `evidence_status` | `required` |
| Equations | No PDE. Rank occupancy only: 64 cells on 4 ranks. |
| Oracle | Uniform: \(16,16,16,16\). Hotspot: rank 0 holds 40 cells; the remaining 24 split evenly as \(8,8,8\). `exact.py` does not read PoPS output. |
| Domain and boundaries | Abstract 1-d cell set. No spatial mesh, BC, or halo. |
| Parameters | Dimensionless. \(N=64\) cells, \(P=4\) ranks, hotspot \(n_0=40\). |
| Native dimensions | `POPS_NATIVE_DIM=1` listed for the planner. Optional `run_native` refuses without a public standalone balancer. |
| Required capabilities | None on the in-memory path. `run_native` raises `NativeUnavailable` (`public standalone load balancer is not available`). |
| Configurations | Two occupancy vectors. No AMR, no resolution series, no timed PF run in this increment. |
| Diagnostics | Per-vector max, mean, and population CV \(\sigma/\mu\). |
| Thresholds | Uniform CV = 0. Hotspot max > mean. |
| Proves | Even occupancy has zero CV. A rank-0 hotspot of 40 cells has max > mean. Report renderer accepts a PF-09 summary with empty `orders`. `run_native` names the missing standalone balancer. |
| Does not prove | Native SFC / knapsack rebalance, MPI migration cost, AMR weights, GPU kernels, timed cells/s. This is not a timed PF run. |
| Resources | Local in-memory contract. No live ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
