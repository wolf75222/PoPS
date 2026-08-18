# PF-12 — HyQMOM15 large-moment state stand-in

Phase 1 kernel-microbench stand-in. A synthetic 15-component state with
HyQMOM15 width, then \(a=2b\) (SAXPY onto a zero destination) on an \((n,15)\)
array. Bytes/cell is \(15\times 8=120\), compared to 5-component Euler width
(\(40\)). In-memory only; no live runtime, compile, or timed PF run.

| Field | Content |
|---|---|
| Identifier | `PF-12` |
| `verification_kind` | `infrastructure` |
| `evidence_status` | `required` |
| Equations | No PDE. Kernel identity: \(a=2b\) on a 15-wide float64 state. |
| Oracle | Unique interior pattern \(b_{i,c}=(i+1)(c+1)\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Width is a state layout, not a mesh. |
| Parameters | Dimensionless. Default \(N=16\) cells. \(15\) HyQMOM15 components. \(5\) Euler components. Scale \(\alpha=2\). Float64 scalar is \(8\) bytes. |
| Native dimensions | `POPS_NATIVE_DIM=1` listed for the planner. Optional `run_native` refuses without a public 15-component Case. |
| Required capabilities | None on the in-memory path. `run_native` raises `NativeUnavailable` (`public 15-component HyQMOM15 Case is not available`). |
| Configurations | Single 1-d stand-in. No AMR, no resolution series, no timed PF run in this increment. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ of \(a\) versus \(2b\). Bytes/cell versus 5-component Euler. |
| Thresholds | SAXPY L1 = L2 = L∞ = 0. HyQMOM15 bytes/cell = 120. Euler bytes/cell = 40. Ratio = 3. |
| Proves | A 15-wide float64 state occupies 120 bytes/cell (3× Euler). \(a=2b\) is exact on \((n,15)\). Report renderer accepts a PF-12 summary with empty `orders`. `run_native` names the missing 15-component Case. |
| Does not prove | Native HyQMOM15, realizability, closure, MultiFab arithmetic, MPI, GPU kernels, timed cells/s, AMR, Poisson, coupling. This is not a timed PF run. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
