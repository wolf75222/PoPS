# PF-02 — scalar MG residual stand-in

Phase 0 kernel stand-in plus optional Phase 7 timing. 1-d residual of
\(-\Delta\phi=\rho\) for the PO-01 trigonometric mode. "V-cycles" are
counted as repeated residual applications of a damped Jacobi stand-in
(\(N=4\)), not native GeometricMG. In-memory identities stay the required
path. Optional `run_native` times PO-01
`verification/cases/poisson/periodic_trig/run.py` and returns
`{elapsed_s, residual_or_error, cells_per_second}`.

| Field | Content |
|---|---|
| Identifier | `PF-02` |
| `verification_kind` | `infrastructure` |
| `evidence_status` | `required` |
| Equations | \(-\phi''=\rho\) on a 1-d periodic interval. Manufactured \(\phi=\sin(2\pi x)\), \(\rho=(2\pi)^2\phi\). Discrete operator is the second-order periodic stencil for \(-\Delta\). Stand-in update is damped Jacobi \(\phi\leftarrow\phi-\omega D^{-1}(A\phi-\rho)\) with \(\omega=2/3\), \(D=2/h^2\). |
| Oracle | Same 1-d PO-01 identities. Residual of analytic \(\phi\) versus manufactured \(\rho\) is the finite-difference truncation of \(-\Delta\), not a solver residual. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Samples are uniform cell centers. |
| Parameters | Dimensionless. Default \(N=32\) cells. Stand-in V-cycle count \(N=4\). Damping \(\omega=2/3\). Cold start \(\phi=0\). |
| Native dimensions | `POPS_NATIVE_DIM=1`. Optional `run_native` reuses PO-01's 1-d FFT Case. |
| Required capabilities | None on the in-memory path. Optional `run_native` needs the PO-01 compiler/Kokkos path. |
| Configurations | Single 1-d stand-in. Optional timed wrap of PO-01 `run_native`. No AMR hierarchy. |
| Diagnostics | Volume-weighted L2 of \(A\phi-\rho\) before and after four residual applications. Analytic-\(\phi\) residual versus the discrete sine eigenvalue \((4/h^2)\sin^2(\pi h)-(2\pi)^2\). |
| Thresholds | Residual L2 after four iterations is strictly smaller than the cold-start residual. Analytic residual matches FD truncation (not machine zero). |
| Proves | Four damped Jacobi residual applications reduce the 1-d PO-01 residual. The analytic potential residual is the stencil truncation. Report renderer accepts a PF-02 summary with empty `orders`. Optional `run_native` times PO-01 when the compiler is present. |
| Does not prove | Native GeometricMG V-cycles, restriction/prolongation, 2-d/3-d Poisson, AMR, MPI, GPU kernels. `elapsed_s` includes PO-01 compile+bind+solve. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
