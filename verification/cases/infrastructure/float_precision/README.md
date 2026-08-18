# IF-09 — float32 vs float64 plateau

Phase 2 infrastructure contract. Reuses the TR-01 manufactured sine and
stores it as `float32` and `float64`. Field-to-field \(L^\infty\) is
\(O(10^{-7})\). Both fields remain finite. In-memory f32/f64 stay the
oracle. Phase 6 `run_native` refuses a float32 Case:
`refuse_float32_case_authoring()` returns
`public float32 Case authoring not active`. Public runtime facts set
`supports_single_precision` to false; `validate_precision("float32")`
refuses. There is no public float32 Case authoring.

| Field | Content |
|---|---|
| Identifier | `IF-09` |
| `verification_kind` | `infrastructure` |
| `evidence_status` | `required` |
| Equations | \(\partial_t q + a\partial_x q = 0\). Conservative scalar \(q\). Canonical \(a=1\). Same IC as TR-01: \(q=q_0+\varepsilon\sin(2\pi k x)\). |
| Oracle | Reused TR-01 manufactured sine via `load_sibling_module` on `verification/cases/transport/advection_sine/exact.py`. Stored as `float32` and `float64`. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Samples are interior cell centers. |
| Parameters | Dimensionless. Default in-memory grid \(N=32\). \(t=0.25\). TR-01 defaults \(q_0=1\), \(\varepsilon=10^{-2}\), \(k=1\), \(a=1\). |
| Native dimensions | `POPS_NATIVE_DIM=1` listed for the planner. `run_native` does not load a native artifact. |
| Required capabilities | None on the in-memory path. Native float32 requires `float32_case_authoring` (currently false). |
| Configurations | Single manufactured field stored in two IEEE dtypes. No solver, CFL, integrator, flux, reconstruction, or AMR. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ of the `float32` field vs the `float64` field. Finiteness of both arrays. |
| Thresholds | \(0 < L^\infty(\mathrm{f32},\mathrm{f64}) \le 10^{-6}\) (\(O(10^{-7})\)). Both fields finite. Empty `orders` with reason `float32 vs float64 plateau / no live compile`. |
| Proves | The TR-01 sine remains finite in `float32` and differs from `float64` by a single-precision plateau. Report renderer accepts an IF-09 summary. `run_native` names the missing float32 Case authoring. |
| Does not prove | Live `POPS_REAL_TYPE=float` compile, a public float32 Case, mixed-precision kernels, native solver plateau, spatial/temporal order, AMR, Poisson, coupling, MPI, performance. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
