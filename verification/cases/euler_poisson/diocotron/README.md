# CP-11 — linear diocotron mode

Cartesian unit-square stand-in of a hollow density ring plus an azimuthal
\(m=2\) perturbation that grows at a documented toy rate \(\gamma=0.1\).
This is an in-memory oracle, not a published-dispersion reproduction.
Native compile is optional.

| Field | Content |
|---|---|
| Identifier | `CP-11` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | 2-d pressureless electron fluid on the unit square. Background ring \(n(r)=n_0\) for \(r_1<r<r_2\), else \(0\). Documented perturbation \(\varepsilon\Re(e^{im\theta}e^{\gamma t})\) with integer mode \(m=2\). Poisson / \(E\times B\) stay on the oracle. |
| Oracle | \(n=n_{\mathrm{ring}}(r)+\varepsilon e^{\gamma t}\cos(m\theta)\). Toy \(\gamma=0.1\) (not a paper growth rate). Unperturbed ring is independent of \(\theta\). Angular FFT of the perturbation peaks at bin \(2\). Amplitude \(\varepsilon e^{\gamma t}\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 2-d periodic unit square \([0,1]^2\), ring centre \((0.5,0.5)\). 1-d is not applicable. |
| Parameters | Dimensionless. \(n_0=1\), \(r_1=0.15\), \(r_2=0.35\), \(\varepsilon=10^{-4}\), \(\gamma=0.1\), \(m=2\). Default \(N=32^2\). |
| Native dimensions | `POPS_NATIVE_DIM=2`. In-memory path does not load a native artifact. |
| Required capabilities | Cartesian 2-d, uniform, periodic, KokkosSerial, MPI off. |
| Configurations | Single resolution \(n=32^2\) for the in-memory report. Authored scheme: Rusanov, MUSCL/VanLeer, SSPRK2, AdaptiveCFL. Poisson and the growth rate stay on the oracle. |
| Diagnostics | Unperturbed ring independent of \(\theta\). Angular rFFT peak at bin \(2\). Amplitude \(\varepsilon e^{\gamma t}\) at sampled times. Task 2 volume-weighted L1/L2/L∞ on density of exact vs exact. |
| Thresholds | Exact-vs-exact L∞ = 0. Amplitude mismatch \(\le 10^{-12}\). Empty `orders` with reason `linear growth / not a published reproduction`. No spatial-order gate on the in-memory path. |
| Proves | Axisymmetric hollow-ring background; \(m=2\) angular content; documented toy exponential growth; report renderer accepts empty orders for a linear-growth stand-in. |
| Does not prove | Native \(\gamma_{\mathrm{num}}\), Davidson / Levy / Rosenthal diocotron dispersion, finite-Larmor or warm-fluid corrections, AMR, MPI, polar mesh, published-paper reproduction. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
