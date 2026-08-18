# GE-04 — Same radial oracle in Cartesian vs polar

Shared Gaussian ring \(\varphi=\exp(-(r-0.5)^2/\sigma^2)\) with \(\sigma=0.08\),
sampled on a Cartesian mesh and on a polar \((r,\theta)\) mesh. Polar samples
are interpolated onto Cartesian cell centres (bilinear; nearest is also
provided). In-memory only; no live polar System, compile, or ROMEO. The
public polar System is not active: `refuse_public_polar_runtime()` returns
`public polar System not active`, and `run_native` raises
`NativeUnavailable` with that string. There is no public PolarMesh runtime.

| Field | Content |
|---|---|
| Identifier | `GE-04` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `capability-gated` |
| Equations | Static scalar field. No evolution. The authored Cartesian Case is stationary 2-d scalar advection (velocity 0) and is not run. |
| Oracle | \(\varphi=\exp(-(r-0.5)^2/\sigma^2)\), \(\sigma=0.08\), \(r=\sqrt{x^2+y^2}\). Polar samples interpolate onto Cartesian (bilinear default). `exact.py` does not read PoPS output. |
| Domain and boundaries | Cartesian box \([-1,1]^2\). Polar disk \(r\in[0,\sqrt{2}]\), \(\theta\in[0,2\pi)\), covering the box. Polar \(\theta\) is periodic. Native polar ends are not active. |
| Parameters | Dimensionless. Default Cartesian \(n=32^2\). Polar \(n_r=64\), \(n_\theta=128\). Ring at \(r=0.5\). |
| Native dimensions | `POPS_NATIVE_DIM=2`. This increment does not load a native artifact. |
| Required capabilities | Cartesian 2-d, uniform, KokkosSerial, MPI off. Polar System runtime is **not** current (`polar_system_runtime = false`). |
| Configurations | Single resolution \(n=32^2\) for the in-memory report. Polar branch stays capability-gated in `case.toml`. Authored Cartesian scheme: ScalarUpwind, MUSCL/VanLeer, SSPRK2, AdaptiveCFL. `run_native` does not compile a polar mesh. |
| Diagnostics | Peak radius of each sampling. Task 2 L1/L2/L∞ of polar→Cartesian interpolation vs Cartesian \(\varphi\). Polar runtime refusal string. |
| Thresholds | Both samplings peak at \(r=0.5\) within one mesh spacing. Field-to-field \(L^\infty<0.05\) on \(32\times 32\) (bilinear). Empty `orders` with reason `capability-gated polar runtime`. |
| Proves | The same radial oracle peaks at \(r=0.5\) on both meshes; bilinear polar→Cartesian interpolation stays under the documented \(L^\infty\) bound; the polar branch remains capability-gated. |
| Does not prove | Live polar runtime, observed spatial order, AMR, Poisson, coupling, MPI, native Cartesian/polar parity. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
