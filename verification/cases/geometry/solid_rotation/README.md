# GE-02 — Solid-body scalar rotation

Polar solid-body rotation of a scalar Gaussian bump, evaluated in Cartesian
coordinates. \(v_\theta=\Omega r\) with \(\Omega=2\pi\) so \(T=1\). The bump
starts at \((0.5,0)\) on the ring \(r=0.5\) and returns to the IC at \(t=T\).
In-memory only; no live runtime, compile, or ROMEO. The public polar System
is not active: `refuse_public_polar_runtime()` returns
`public polar System not active`, and `run_native` raises
`NativeUnavailable` with that string. There is no public PolarMesh runtime.

| Field | Content |
|---|---|
| Identifier | `GE-02` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `capability-gated` |
| Equations | \(\partial_t q + \nabla\cdot(\mathbf{u} q)=0\) with \(\mathbf{u}=(-\Omega y,\Omega x)\). Conservative scalar \(q\). No sources. |
| Oracle | \(v_\theta=\Omega r\), \(v_r=0\), \(\Omega=2\pi\), \(T=1\). Cartesian evaluation \(x=r\cos\theta\), \(y=r\sin\theta\). Compact Gaussian of width \(\sigma=0.08\) centred at \((0.5,0)\). Oracle at \(t=T\) is the IC (`exact_return`). `exact.py` does not read PoPS output. |
| Domain and boundaries | 2-d box \([-1,1]^2\), rotation about the origin. Native non-periodic ends stay later; the shared layout helper is periodic. |
| Parameters | Dimensionless. \(\Omega=2\pi\). Ring radius \(r=0.5\). Default in-memory grid \(n=32^2\). |
| Native dimensions | `POPS_NATIVE_DIM=2`. This increment does not load a native artifact. |
| Required capabilities | Cartesian 2-d, uniform, KokkosSerial, MPI off. Polar mesh runtime is not active (`polar_system_runtime = false`); the case is capability-gated on that path. |
| Configurations | Single resolution \(n=32^2\) for the in-memory report. Authored Cartesian scheme: ScalarUpwind, MUSCL/VanLeer, SSPRK2, AdaptiveCFL. Manufactured rotation lives in `exact.py`. `run_native` does not compile a polar mesh. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ of `exact_return` vs the IC. Quarter-period peak location after a 90° rotation. |
| Thresholds | Return L1 = L2 = L∞ = 0. At \(t=T/4\) the peak is at \((0,0.5)\) (grid peak within one cell). No spatial-order gate on the in-memory path. |
| Proves | Exact return at \(t=T\) equals the IC. The bump rotates 90° in a quarter period. Cartesian samples of \(v_\theta=\Omega r\) are \((u,v)=(-\Omega y,\Omega x)\). Report renderer accepts a GE-02 summary. |
| Does not prove | Native 2-d advection order, polar-mesh runtime, AMR, Poisson, coupling, MPI, performance. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
