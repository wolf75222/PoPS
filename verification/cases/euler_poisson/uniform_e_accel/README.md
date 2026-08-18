# CP-08 — uniform-E acceleration

1-d Euler–Poisson species in a prescribed uniform electric field. Spatial
dynamics are absent: density stays uniform, velocity is spatially uniform.
Native compile is optional.

| Field | Content |
|---|---|
| Identifier | `CP-08` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | One species, charge \(q\), mass \(m\). Momentum source \(qnE_0\) with prescribed uniform \(E=E_0\). Continuity has no spatial flux when \(n\) and \(u\) are uniform. Conserved \((n, nu, \mathcal{E})\) with \(\rho=nm\). |
| Oracle | \(u(t)=u_0+(q/m)E_0 t\). \(n(x,t)=n_0\). Kinetic energy density \(\frac12 n_0 m u(t)^2\). Opposite charges have opposite accelerations. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). |
| Parameters | Dimensionless defaults \(q=1\), \(m=1\), \(E_0=1\), \(u_0=0\), \(n_0=1\), \(p_0=1\), \(\gamma=1.4\). |
| Native dimensions | `POPS_NATIVE_DIM=1`. In-memory path does not load a native artifact. |
| Required capabilities | Cartesian 1-d, uniform, periodic, KokkosSerial, MPI off. Rusanov + MUSCL when a native series exists. Poisson solve is not required: \(E_0\) is prescribed. |
| Configurations | Single resolution \(n=32\) for the in-memory report. Authored scheme: Rusanov, MUSCL/VanLeer, SSPRK2, AdaptiveCFL. Lorentz source is exported by the oracle and is not yet injected into the Case. |
| Diagnostics | Linear \(u(t)\). Sign of \(q\). Kinetic-energy formula. Task 2 volume-weighted L1/L2/L∞ on density of exact vs exact. |
| Thresholds | Exact-vs-exact L∞ = 0. No spatial-order gate on the in-memory path. |
| Proves | Closed uniform-E oracle (linear velocity; opposite charges accelerate oppositely; density unchanged; exact KE); report renderer. |
| Does not prove | Observed temporal order, native Lorentz-source injection, implicit sources, AMR, Poisson solve, MPI, multi-d, FE/RK3 series. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
