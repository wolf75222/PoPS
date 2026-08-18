# PF-04 — Euler Rusanov / Lax-Friedrichs step

Phase 0 kernel stand-in plus optional EU-01 native timing. One first-order
Rusanov (local Lax-Friedrichs) update of a uniform 1-d Euler free stream
must remain that stream. The python face loop is timed only as an
observation. Optional ``run_native`` times EU-01
``verification/cases/euler/linear_waves/run.py`` and returns elapsed plus
cells/s. GPU spaces are refused.

| Field | Content |
|---|---|
| Identifier | `PF-04` |
| `verification_kind` | `infrastructure` |
| `evidence_status` | `required` |
| Equations | 1-d gamma-law Euler, primitives \(W=(\rho,u,p)\). Conserved \((\rho,\rho u,E)\). No sources. \(\gamma=1.4\). First-order finite-volume Rusanov / local Lax-Friedrichs: \(\hat F=\tfrac12(F_L+F_R)-\tfrac12\alpha(U_R-U_L)\), \(\alpha=\max(\lvert u\rvert+c)_L^R\). |
| Oracle | Spatially constant free stream \((\rho,u,p)=(1,1,1)\). Exact at every \(t\) is the IC. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). |
| Parameters | Dimensionless. Default \(N=32\) cells. CFL \(0.4\). \(\gamma=1.4\). Free-stream L∞ tolerance \(10^{-15}\). |
| Native dimensions | `POPS_NATIVE_DIM=1` listed for the planner. Optional `run_native` reuses EU-01 Dim1. |
| Required capabilities | None on the in-memory path. Native timing needs EU-01 compile (Kokkos). No public CUDA space. |
| Configurations | Single 1-d stand-in. One forward-Euler Rusanov step. Optional EU-01 timing. No AMR, no GPU, no two-node. |
| Diagnostics | Primitive L∞ of \((\rho,u,p)\) versus the uniform stream after one step. Observed python-loop cells/s (not a gate). Optional EU-01 `elapsed_s` / `cells_per_second`. |
| Thresholds | After one step, \(\max\lvert\rho-\bar\rho\rvert\), \(\max\lvert u-\bar u\rvert\), \(\max\lvert p-\bar p\rvert\) \(\le 10^{-15}\). |
| Proves | A uniform Euler state is a discrete free-stream fixed point of one periodic Rusanov / LF step. Report renderer accepts a PF-04 summary with empty `orders` and a python-loop observation. `run_native` times EU-01 or refuses GPU / missing Kokkos. |
| Does not prove | A standalone native Rusanov kernel, MUSCL/HLLC, geometric conservation on a moving mesh, AMR, Poisson, coupling, MPI, GPU kernels, or a two-node cells/s claim. Native elapsed includes EU-01 compile+bind+run. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
