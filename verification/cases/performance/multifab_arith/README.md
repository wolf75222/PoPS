# PF-01 — MultiFab arithmetic / periodic halo

Phase 0 kernel-microbench stand-in plus an optional Phase 7 native timer. A
1-d numpy field with a periodic halo of width 2, then \(a=b+c\) and \(a=2b\).
Halo cells must match the wrapped interior after fill. In-memory oracles stay
the contract. Optional ``run_native`` times TR-01
(`verification/cases/transport/advection_sine/run.py`) and does not call
``pops.run`` here.

| Field | Content |
|---|---|
| Identifier | `PF-01` |
| `verification_kind` | `infrastructure` |
| `evidence_status` | `required` |
| Equations | No PDE. Kernel identities: periodic wrap of width 2; \(a=b+c\); \(a=2b\) (SAXPY onto a zero destination). |
| Oracle | Unique interior pattern \(b_i=i+1\). Partner \(c_i=3i+0.5\). Halo oracle is the periodic wrap of the interior. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Valid cells plus a halo of width 2 on each side. |
| Parameters | Dimensionless. Default \(N=16\) interior cells. Halo width 2. Scale \(\alpha=2\). |
| Native dimensions | `POPS_NATIVE_DIM=1`. Optional `run_native` reuses TR-01 Dim1 compile/bind/run. |
| Required capabilities | None on the in-memory path. The optional timer needs TR-01 native compile (MUSCL/VanLeer + ScalarUpwind, SSPRK2, KokkosSerial). |
| Configurations | Single 1-d stand-in. Optional TR-01 native timer. No AMR, no isolated MultiFab kernel series. |
| Diagnostics | Halo L∞ versus the periodic interior wrap. Task 2 volume-weighted L1/L2/L∞ of \(a\) versus \(2b\) and versus \(b+c\). |
| Thresholds | Halo L∞ = 0. SAXPY L1 = L2 = L∞ = 0. Add L1 = L2 = L∞ = 0. |
| Proves | Periodic halo fill of width 2 copies the wrapped interior. \(a=2b\) and \(a=b+c\) are exact on valid cells. Report renderer accepts a PF-01 summary with empty `orders`. Optional `run_native` returns `{elapsed_s, n_cells, n_steps, cells_per_second, field}` from a timed TR-01 advance. |
| Does not prove | Isolated native MultiFab `saxpy`/`lincomb` or `fill_boundary`. The timer is a TR-01 compile+advance proxy, not a MultiFab microbench. No MPI, GPU, or two-node claim. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
