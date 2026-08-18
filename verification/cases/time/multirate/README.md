# TM-06 — multirate species substeps

P1 temporal contract. Two uncoupled linear decays \(y'=-\lambda_f y\),
\(z'=-\lambda_s z\) with \(\lambda_f=8\), \(\lambda_s=1\). Fast species
takes \(r\in\{1,2,4,8\}\) backward-Euler substeps of size \(\Delta t/r\);
slow species takes one BE step of size \(\Delta t\). In-memory BE remains
the oracle. A public 1-cell 1-d periodic Case authors the same pair as
local linear operators and a Program with a child clock plus subcycle.
`pops.lib.time` has IMEX/BDF, not a Multirate factory. Native compile is
optional and does not call ROMEO.

| Field | Content |
|---|---|
| Identifier | `TM-06` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | \(y'=-\lambda_f y\), \(z'=-\lambda_s z\). Uncoupled. Backward Euler: \(u\mapsto u/(1+\lambda\Delta t)\). Multirate: \(r\) fast substeps of \(\Delta t/r\), one slow step of \(\Delta t\). |
| Oracle | Closed form \(y(t)=y_0 e^{-\lambda_f t}\), \(z(t)=z_0 e^{-\lambda_s t}\). `exact.py` does not read PoPS output. |
| Domain and boundaries | In-memory path: 0-d / two-component ODE. Public Case: 1-d periodic unit interval, 1 cell. |
| Parameters | Dimensionless. Canonical \(\lambda_f=8\), \(\lambda_s=1\), \(y_0=z_0=1\), \(\Delta t=0.25\). Substep ratios \(r=1,2,4,8\). |
| Native dimensions | `POPS_NATIVE_DIM=1` only. Selected by the public resolve/compile path. |
| Required capabilities | None on the in-memory path. Public Case: Cartesian uniform periodic, child `Clock` + `Program.subcycle` + local-linear BE + `FixedDt`. KokkosSerial. MPI off. |
| Configurations | Multirate BE vs single-rate BE. No AMR, no \(\Delta t\) series in this increment. |
| Diagnostics | Exact exponential identity. \(r=1\) identity with single-rate BE. Fast-component error vs the exact exponential at fixed \(\Delta t\). |
| Thresholds | \(r=1\) matches single-rate BE exactly. Fast-component error is strictly decreasing in \(r\) at the documented \(\Delta t\). |
| Proves | Closed-form two-rate exponential. \(r=1\) recovers single-rate BE. Larger \(r\) reduces the fast-component error at fixed \(\Delta t\). Public two-species Case resolves in Dim1 without compile. Report renderer accepts a TM-06 summary. |
| Does not prove | Native multirate Program order until a compiler series is run. Coupled species, IMEX/Strang, AMR, Poisson, MPI, performance. |
| Resources | Local in-memory contract. Public Case is 1-d, 1 cell. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compile records stay on the optional `run_native` path. |
