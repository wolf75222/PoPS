# EU-03 — Euler manufactured solution

1-d reduction of a time-dependent gamma-law Euler MMS. Oracle is the
manufactured primitive field plus the conservative residual
\(S=\partial_t U+\partial_x F\). Native compile is optional.

| Field | Content |
|---|---|
| Identifier | `EU-03` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | 1-d gamma-law Euler, primitives \(W=(\rho,u,p)\). Conserved \((\rho,\rho u,E)\). Manufactured source \(S=\partial_t U+\partial_x F\). \(\gamma=1.4\). |
| Oracle | 1-d \(\rho=2+0.1\sin(2\pi(x-t))\), \(u=0.3+0.1\cos(2\pi(x-t))\), \(p=1+0.05\sin(2\pi(x-t))\). 2-d plan fields \(\rho=2+0.1\sin(2\pi(x+y-t))\), \(u=0.3+0.1\cos(2\pi(x-t))\), with plan completion \(v=-0.2+0.1\sin(2\pi(y+t))\), \(p=1+0.1\cos(2\pi(x-y+t))\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). 2-d fields are sampled on the same unit interval in each coordinate. |
| Parameters | \(\gamma=1.4\). Dimensionless. Density stays in \([1.9,2.1]\); 1-d pressure stays in \([0.95,1.05]\). |
| Native dimensions | `POPS_NATIVE_DIM=1`. In-memory path does not load a native artifact. |
| Required capabilities | Cartesian 1-d, uniform, periodic, KokkosSerial, MPI off. HLLC/Rusanov + MUSCL when a native series exists. |
| Configurations | Single resolution \(n=32\) for the in-memory report. Authored scheme: Rusanov, MUSCL/VanLeer, SSPRK2, AdaptiveCFL. Sources are exported by `exact.sources_1d` and are not yet injected into the Case. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ on density of exact vs exact. Positivity of \(\rho,p\). Time-shift identity of the 1-d traveling wave. |
| Thresholds | Exact-vs-exact L∞ = 0. No spatial-order gate on the in-memory path. |
| Proves | 1-d MMS oracle (positivity; exact field is a unit-speed traveling wave); analytic manufactured residual \(S=\partial_t U+\partial_x F\); report renderer. |
| Does not prove | Observed spatial/temporal order, native source injection, AMR, Poisson, coupling, MPI, 2-d/3-d residual, Dirichlet, multi-stage source evaluation. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
