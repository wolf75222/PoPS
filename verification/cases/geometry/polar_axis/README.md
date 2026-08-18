# GE-05 — Polar axis regularity / volume conservation

Polar cell volume \(r\,\Delta r\,\Delta\theta\) on an annulus. A constant
state 1 integrates to the analytic area. The axis cell at \(r=0\) is refused
or given the documented regular volume \(\tfrac12(\Delta r)^2\Delta\theta\).
In-memory only; no live runtime, compile, or ROMEO. The public polar System
is not active: `refuse_public_polar_runtime()` returns
`public polar System not active`, and `run_native` raises
`NativeUnavailable` with that string. There is no public PolarMesh runtime.

| Field | Content |
|---|---|
| Identifier | `GE-05` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `capability-gated` |
| Equations | Polar control-volume area \(V=r\,\Delta r\,\Delta\theta\). Midpoint \(r\) makes this exact: \(\sum V_{ij}=\pi(r_\mathrm{out}^2-r_\mathrm{in}^2)\). |
| Oracle | Uniform annulus \(r\in[0.2,1]\), \(\theta\in[0,2\pi)\), \(n_r=8\), \(n_\theta=16\). Axis helper \(\tfrac12(\Delta r)^2\Delta\theta\) does not divide by \(r=0\). `exact.py` does not read PoPS output. |
| Domain and boundaries | Annulus excluding the axis. \(\theta\) periodic. Native polar ends stay later. |
| Parameters | Dimensionless. Default in-memory grid \(8\times16\). Axis demonstration uses \(\Delta r=1/8\). |
| Native dimensions | `POPS_NATIVE_DIM=2`. This increment does not load a native artifact. |
| Required capabilities | `polar_system_runtime` (currently false). Public polar System is refused. |
| Configurations | Single annulus resolution for the in-memory report. No flux, reconstruction, or time stepper. `run_native` does not compile a polar mesh. |
| Diagnostics | Discrete annulus volume vs \(\pi(r_\mathrm{out}^2-r_\mathrm{in}^2)\). Constant-state integral of 1. Axis-cell regular volume. |
| Thresholds | Volume residual \(\le 10^{-14}\). Axis helper is exactly \(\tfrac12(\Delta r)^2\Delta\theta\). No spatial-order gate. |
| Proves | Midpoint polar volumes conserve the annulus area; axis helper is regular and does not divide by \(r=0\); report renderer. |
| Does not prove | Native polar runtime, polar Poisson, polar hydro, AMR, MPI, observed spatial order. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
