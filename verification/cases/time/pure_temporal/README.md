# TM-01 — pure temporal order

Phase 1 temporal-order case. Reuses the TR-01 manufactured sine on a fixed
fine grid and varies \(\Delta t\). Native compile is optional; the in-memory
path manufactures \(E\propto\Delta t^2\) for SSPRK2 and does not call a solver.

| Field | Content |
|---|---|
| Identifier | `TM-01` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | \(\partial_t q + a \partial_x q = 0\). Conservative scalar \(q\). Canonical \(a=1\). Time integrator SSPRK2 (RK2). |
| Oracle | Reused TR-01 manufactured translation \(q(x,t)=q(x-at,0)\) of \(q=q_0+\varepsilon\sin(2\pi k x)\). Loaded with `load_sibling_module` from `verification/cases/transport/advection_sine/exact.py`. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). `POPS_NATIVE_DIM=1`. |
| Parameters | Dimensionless. Fine grid \(N=64\) fixed. Temporal series \(\Delta t,\Delta t/2,\Delta t/4,\Delta t/8\) with \(\Delta t=1/128\). |
| Native dimensions | `POPS_NATIVE_DIM=1` only. Selected by the public resolve/compile path. |
| Required capabilities | Cartesian uniform periodic. KokkosSerial. MPI off. MUSCL/VanLeer + ScalarUpwind, SSPRK2 + `FixedDt`. |
| Configurations | Uniform 1-d cells. Formal temporal order 2. No AMR. Spatial mesh held fixed. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ vs the manufactured sine. Task 15 observed order on the \(\Delta t\) series (`kind=temporal`). |
| Thresholds | Observed order \(\ge 1.8\) for declared RK2 / SSPRK2 (plan §8.2). Manufactured \(E\propto\Delta t^2\) yields order 2. In-memory exact-vs-exact L∞ is 0. |
| Proves | TR-01 oracle reuse via `load_sibling_module`. Manufactured RK2 temporal order. Public Case authoring resolves in Dim1 without compile. Report renderer accepts a TM-01 temporal summary. |
| Does not prove | Native temporal order until a compiler series is run. FE / RK3 / Strang variants. EU-01 / CP-02 / TM-03 reuse. AMR, Poisson, coupling, MPI, performance. |
| Resources | Local 1-d \(\Delta t\) series on \(N=64\). No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compile records stay on the optional `run_native` path. |
