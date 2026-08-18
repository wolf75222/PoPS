# TM-08 — reversible Strang

Phase 4 temporal reversibility contract. Advance time \(T\) at velocity
\(+a\), flip the velocity sign, and advance \(T\) at \(-a\). The exact
reversible map is the identity, so the return error is 0. A public 1-d
periodic Case authors the same TR-01 sine advection so `resolve_plan`
succeeds. Native compile is optional and does not call ROMEO.

| Field | Content |
|---|---|
| Identifier | `TM-08` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | \(\partial_t q + a \partial_x q = 0\). Conservative scalar \(q\). Canonical \(a=1\). The reversible pair is the exact translation \(+aT\) followed by \(-aT\). |
| Oracle | Reused TR-01 manufactured sine \(q(x,t)=q_0+\varepsilon\sin(2\pi k(x-at))\), loaded with `load_sibling_module` from `verification/cases/transport/advection_sine/exact.py`. The composed departure \(\mathrm{mod}(x-aT-(-a)T,1)\) is \(x\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Samples are uniform cell centers. Public Case uses the same interval. |
| Parameters | Dimensionless. Fine grid \(N=64\). Forward duration \(T=0.25\) (not a full period, so the intermediate state is shifted). Velocity \(a=1\). Public Case takes a `FixedDt` argument. |
| Native dimensions | `POPS_NATIVE_DIM=1` only. Selected by the public resolve/compile path. |
| Required capabilities | None on the in-memory path. Public Case: Cartesian uniform periodic, MUSCL/VanLeer + ScalarUpwind, SSPRK2 + `FixedDt`. KokkosSerial. MPI off. |
| Configurations | Uniform 1-d cells. Exact reversible map on the in-memory path. No AMR, no \(\Delta t\) series in this increment. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ of the returned field vs the initial sine. Forwarded field must differ from the initial field at \(T=0.25\). |
| Thresholds | Return L1 = L2 = L∞ = 0. |
| Proves | TR-01 oracle reuse via `load_sibling_module`. Exact reversible map is the identity. Public Case resolves in Dim1 without compile. Report renderer accepts a TM-08 summary. |
| Does not prove | Native Strang reversibility, discrete time-reversal of a compiled Program, TM-02 noncommuting order, TM-03 collision relaxation, AMR, Poisson, coupling, MPI, performance. |
| Resources | Local in-memory contract. Public Case is 1-d, \(N=64\). No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compile records stay on the optional `run_native` path. |
