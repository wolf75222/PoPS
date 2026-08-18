# AM-02 — prescribed moving patch

Phase 3 AMR contract. The fine patch follows the TR-02 exact barycenter,
independent of any numerical gradient estimator. A public two-level 1-d
periodic AMR Case tags a marker bump advected at the exact speed \(a\) from
\(x_0\). Leftover in-memory helpers remain; `run_native` compiles, binds, and
runs when Kokkos and a compiler are present.

| Field | Content |
|---|---|
| Identifier | `AM-02` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | Scalar advection \(\partial_t q + a\partial_x q = 0\). The patch center is prescribed as the exact pulse barycenter. A marker bump travels at the exact speed \(a\) from \(x_0\); `ValueExpr(marker) > threshold` tags the fine patch. Regrid error is recorded immediately before and after each regrid. |
| Oracle | TR-02 translated Gaussian, loaded with `load_sibling_module`. Closed-form center \((x_0+a t)\bmod 1\). Manufactured regrid jump \(\propto h^2\). Exact-field mass drift over 256 refine/coarsen cycles is the observation 0. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\) via `CartesianDomain` + `Cartesian1D`. |
| Parameters | TR-02 defaults \(x_0=0.37\), \(a=1\). Stress: 256 cycles, \(\Delta t=1/256\). Live regrid every 2 accepted steps. Manufactured jump coefficients are dimensionless. |
| Native dimensions | `POPS_NATIVE_DIM=1`. `run_native` returns the level-0 tracer as shape `(n_cells,)`. |
| Required capabilities | Public AMR (`layouts.AMR`, `AMRHierarchy(max_levels=2, ratios=(2,))`), Cartesian 1-d periodic, SSPRK2 on tracer and marker, live `AMRRegrid(schedule=every(N))`. |
| Configurations | Prescribed patch vs tagging (AM-03, out of scope). No fitted order series in this increment. |
| Diagnostics | Patch center vs \(x_0+a t\); error_before / error_after (two scalars); 256-cycle mass drift (observation). |
| Thresholds | Center matches \(x_0+a t\). Manufactured jump ratio at \(h\) and \(h/2\) is 4. Exact mass drift is 0. |
| Proves | Public 1-d periodic AMR Case validates and resolves. Prescribed center equals the TR-02 exact barycenter. Regrid before/after scalars have a jump \(\propto h^2\). Exact-field 256-cycle mass drift is the observation 0. Report renderer accepts an AM-02 summary. |
| Does not prove | Prolongation/restriction quality, tagging (AM-03), 512-cycle release stress, MPI, performance, fitted spatial order. Native compile requires Kokkos + a compiler. |
| Resources | Local authoring and in-memory contract. Native path is optional. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compile records compiler/Kokkos only when `run_native` runs. |
