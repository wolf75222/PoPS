# TR-03 — reversible SingleVortex

Phase 1 incompressible time-reversible 2-d swirl on the unit square. The
manufactured velocity reverses over one period \(T=1\), so the exact scalar at
\(t=T\) is the initial condition. In-memory only; no live runtime, compile, or
ROMEO.

| Field | Content |
|---|---|
| Identifier | `TR-03` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | \(\partial_t q + \nabla\cdot(\mathbf{u} q)=0\) with divergence-free \(\mathbf{u}=(u,v)\). Conservative scalar \(q\). No sources. |
| Oracle | Time-reversible LeVeque swirl \(u=\sin^2(\pi x)\sin(2\pi y)\cos(\pi t/T)\), \(v=-\sin^2(\pi y)\sin(2\pi x)\cos(\pi t/T)\), \(T=1\). Compact \(\cos^6\) disk centred at \((0.5,0.75)\). Oracle at \(t=T\) is the IC (`exact_return`). `exact.py` does not read PoPS output. |
| Domain and boundaries | 2-d periodic unit square \([0,1]^2\). 1-d is not applicable. |
| Parameters | Dimensionless. \(T=1\). Disk radius \(R=0.15\). Default in-memory grid \(N=32^2\). |
| Native dimensions | `POPS_NATIVE_DIM=2` listed for the planner. This increment does not load a native artifact. |
| Required capabilities | None on the in-memory path. A later native series needs 2-d prescribed incompressible advection. |
| Configurations | Uniform 2-d cells. Exact return-to-IC only. No spatial-order series in this increment. |
| Diagnostics | Discrete central-difference divergence of manufactured \((u,v)\) at cell centres. Task 2 volume-weighted L1/L2/L∞ of `exact_return` vs the IC. |
| Thresholds | \(\max\lvert\nabla_h\cdot\mathbf{u}\rvert < 10^{-2}\) on \(N=32\). Return L1 = L2 = L∞ = 0. |
| Proves | Manufactured velocity is incompressible at cell centres. Exact return at \(t=T\) equals the IC. Report renderer accepts a TR-03 summary. |
| Does not prove | Native 2-d advection order, particle-path integration at intermediate \(t\), AMR, Poisson, coupling, MPI, performance. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
