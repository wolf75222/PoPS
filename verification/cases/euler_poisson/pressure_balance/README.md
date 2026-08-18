# CP-07 — pressure–field equilibrium

Phase 2 code-verification case. In-memory isothermal Euler–Poisson
hydrostatic oracle only. Public 1-d Euler Case authoring is present. No
native compile, no bind, no `pops.run`, no ROMEO.

| Field | Content |
|---|---|
| Identifier | `CP-07` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | Isothermal Euler–Poisson rest state. \(p=Tn\), \(u=0\), Boltzmann \(\phi=-(T/q)\ln n+C\), \(E=-d\phi/dx\). Force balance \(\nabla p=qnE\). |
| Oracle | 1-d \(n=n_0(1+\delta\cos(2\pi x))\) with \(\delta<1\), or \(n=n_0\exp(-x^2/(2\sigma^2))\). Defaults \(n_0=1\), \(T=1\), \(q=-1\), \(\delta=0.1\), \(\sigma=0.2\), \(C=0\). `exact.py` does not read PoPS output. |
| Domain and boundaries | Cosine: periodic unit interval \([0,1]\). Gaussian: sampled on \([-1/2,1/2]\). |
| Parameters | Dimensionless \(e=1\). Density stays positive (\(\delta<1\)). |
| Native dimensions | `POPS_NATIVE_DIM=1`. In-memory path does not load a native artifact. |
| Required capabilities | Cartesian 1-d, uniform, periodic, KokkosSerial, MPI off. Coupled Euler–Poisson native solve is out of scope. |
| Configurations | Single resolution \(n=64\) for the in-memory report. Authored hydro scheme: Rusanov, MUSCL/VanLeer, SSPRK2, AdaptiveCFL. Poisson / force balance live in the oracle. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ of exact vs exact density / \(\phi\) / \(E\). Force-balance residual \(\|\nabla p-qnE\|_\infty\). Rest velocity. Coupling `sign_ok`. |
| Thresholds | Exact-vs-exact L∞ = 0. Force-balance residual \(\le 10^{-12}\). No spatial-order gate on the in-memory path. |
| Proves | Isothermal Boltzmann identity; \(\nabla p=qnE\) on both exact profiles; \(u=0\); schema-valid campaign report. |
| Does not prove | Native Euler–Poisson solve, observed spatial/temporal order, AMR, MPI, Langmuir dynamics, multi-d. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
