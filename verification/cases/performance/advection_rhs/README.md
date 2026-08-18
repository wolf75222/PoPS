# PF-03 — Advection FV RHS / periodic halo

Phase 0 kernel-microbench stand-in plus an optional Phase 7 native timer. A
1-d numpy field with a periodic halo of width 1, then the first-order upwind
finite-volume RHS of \(q_t + a q_x\) on the TR-01 sine. Halo fill is split from
the interior stencil. In-memory oracles stay the contract. Optional
``run_native`` times TR-01 (`verification/cases/transport/advection_sine/run.py`)
and does not call ``pops.run`` here.

| Field | Content |
|---|---|
| Identifier | `PF-03` |
| `verification_kind` | `infrastructure` |
| `evidence_status` | `required` |
| Equations | \(\partial_t q + a \partial_x q = 0\). Conservative scalar \(q\). Canonical \(a=1\). Stand-in kernel is first-order upwind: \(\mathrm{RHS}_i = -a(q_i-q_{i-1})/\Delta x\) after a periodic wrap of width 1. |
| Oracle | TR-01 manufactured sine \(q=q_0+\varepsilon\sin(2\pi k(x-at))\) with \(q_0=1\), \(\varepsilon=10^{-2}\), \(k=1\). Exact translation RHS is \(-a\partial_x q=-a\,\varepsilon\,2\pi k\cos(2\pi k(x-at))\) at cell centres. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Valid cells plus a halo of width 1 on each side. |
| Parameters | Dimensionless. Default \(N=16\) interior cells. Halo width 1. Speed \(a=1\). |
| Native dimensions | `POPS_NATIVE_DIM=1`. Optional `run_native` reuses TR-01 Dim1 compile/bind/run. |
| Required capabilities | None on the in-memory path. The optional timer needs TR-01 native compile (MUSCL/VanLeer + ScalarUpwind, SSPRK2, KokkosSerial). |
| Configurations | Single 1-d stand-in. Optional TR-01 native timer. No AMR, no isolated RHS kernel series. |
| Diagnostics | Halo L∞ versus the periodic interior wrap. Task 2 volume-weighted L1/L2/L∞ of the interior RHS versus the analytic \(-a\partial_x\) sine. |
| Thresholds | Halo L∞ = 0. Interior RHS L∞ \(\le |a|\,\varepsilon\,(2\pi k)^2\Delta x\) (first-order FD bound). |
| Proves | Periodic halo fill of width 1 copies the wrapped interior. Interior first-order upwind RHS matches \(-a\partial_x\) sine at centres to FD order. Report renderer accepts a PF-03 summary with empty `orders`. Optional `run_native` returns `{elapsed_s, n_cells, n_steps, cells_per_second, field}` from a timed TR-01 advance. |
| Does not prove | Isolated native FV upwind RHS or `fill_boundary`. The timer is a TR-01 compile+advance proxy (MUSCL/VanLeer), not a first-order RHS microbench. No MPI, GPU, or two-node claim. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
