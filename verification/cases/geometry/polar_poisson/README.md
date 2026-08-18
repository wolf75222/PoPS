# GE-01 — Polar manufactured Poisson (harmonic on an annulus)

Manufactured harmonic \(\varphi=r^{m}\cos(m\theta)\) on the annulus
\(r\in[0.2,1]\), \(m=2\). Polar Laplacian \(\Delta\varphi=0\). Cartesian
equivalent \(\operatorname{Re}((x+iy)^{m})\). In-memory only. The public
polar System is not active: `refuse_public_polar_runtime()` returns
`public polar System not active`, and `run_native` raises
`NativeUnavailable` with that string. There is no public PolarMesh runtime.

| Field | Content |
|---|---|
| Identifier | `GE-01` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `capability-gated` |
| Equations | Polar Poisson \(-\Delta\varphi=f\) with \(f=0\) for this harmonic. Polar Laplacian \(\Delta=\partial_{rr}+r^{-1}\partial_{r}+r^{-2}\partial_{\theta\theta}\). |
| Oracle | \(\varphi(r,\theta)=r^{m}\cos(m\theta)\), \(m=2\). `polar_laplacian` is identically 0 on \(r>0\). `cartesian_equivalent` is \(\operatorname{Re}((x+iy)^{m})\). `exact.py` does not read PoPS output. |
| Domain and boundaries | Annulus \(r\in[0.2,1]\), \(\theta\in[0,2\pi)\). Origin \(r=0\) is excluded. Authored Cartesian box \([-1,1]^{2}\) is Dirichlet for the equivalent. |
| Parameters | Dimensionless. \(m=2\), \(r_{\min}=0.2\), \(r_{\max}=1\). Default in-memory polar grid \(32\times 64\). |
| Native dimensions | `POPS_NATIVE_DIM=2`. This increment does not load a native artifact. |
| Required capabilities | Polar Poisson runtime is not public (`polar_system_runtime = false`). Cartesian 2-d Poisson authoring is present. MPI off. |
| Configurations | Single in-memory polar sample. Authored Cartesian scheme: GeometricMG, cell-centered second order, Dirichlet. Polar System authoring is refused. `run_native` does not compile a polar mesh. |
| Diagnostics | Analytic polar Laplacian at sample \((r,\theta)\). Origin excluded from the annulus. `refuse_public_polar_runtime()` reason string. Task 2 exact-vs-exact L1/L2/L∞ of Cartesian vs polar \(\varphi\). |
| Thresholds | \(\lvert\Delta\varphi\rvert\le 10^{-12}\). Exact-vs-exact L∞ = 0. Refuse helper non-empty. No spatial-order gate (`capability-gated polar runtime`). |
| Proves | The manufactured harmonic is Laplacian-free on the annulus; the Cartesian equivalent matches \(\varphi(r,\theta)\); \(r=0\) is excluded; the public polar System refusal is documented; report renderer. |
| Does not prove | Observed spatial order, live polar Poisson solve, PolarPoissonSolver FFT/Thomas, AMR, coupling, MPI, native polar System. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
