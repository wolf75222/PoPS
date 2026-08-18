# AM-05 — regrid frequency sweep

Phase 3 AMR cadence contract. The same prescribed TR-02 pulse is advanced
for \(N=16\) steps while the tagged window is rebuilt every
\(k\in\{1,2,4,8,16\}\) steps of \(\Delta t\). Leftover in-memory helpers
remain. A public two-level 1-d periodic AMR Case tags a marker bump
advected at the exact speed \(a\) from \(x_0\), with
`AMRRegrid(schedule=every(k, clock=program.clock))`. `run_native`
compiles, binds, and runs when Kokkos and a compiler are present.

| Field | Content |
|---|---|
| Identifier | `AM-05` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | Scalar advection \(\partial_t q + a\partial_x q = 0\). The tagged window is prescribed around the exact pulse barycenter and rebuilt every \(k\) steps. A marker bump travels at the exact speed \(a\) from \(x_0\); `ValueExpr(marker) > threshold` tags the fine patch. |
| Oracle | TR-02 translated Gaussian, loaded with `load_sibling_module`. Rebuild count is \(N/k\). Leftover cadence \(\lvert k_{\mathrm{regrid}}-k_{\mathrm{requested}}\rvert=0\). Exact field is independent of \(k\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\) via `CartesianDomain` + `Cartesian1D`. |
| Parameters | TR-02 defaults \(x_0=0.37\), \(a=1\). \(N=16\), \(\Delta t=1/16\). Frequencies \(k\in\{1,2,4,8,16\}\). Window half-width \(4\sigma\). Live default `regrid_every=2`. |
| Native dimensions | `POPS_NATIVE_DIM=1`. `run_native` returns the level-0 tracer as shape `(n_cells,)`. |
| Required capabilities | Public AMR (`layouts.AMR`, `AMRHierarchy(max_levels=2, ratios=(2,))`), Cartesian 1-d periodic, SSPRK2 on tracer and marker, live `AMRRegrid(schedule=every(k))`. |
| Configurations | Frequency sweep \(k\in\{1,2,4,8,16\}\). No tagging (AM-03), interface-band (AM-01), or fitted order series in this increment. |
| Diagnostics | Rebuild count vs \(N/k\); leftover \(\lvert k_{\mathrm{regrid}}-k_{\mathrm{requested}}\rvert\); L∞ leftover of the exact field vs \(1/k\). |
| Thresholds | Rebuilds \(=N/k\). Interval leftover \(=0\). Exact-field leftover slope vs \(1/k\) is \(0\). |
| Proves | Public 1-d periodic AMR Case validates and resolves for at least \(k=2\) and \(k=4\). The tagged window is rebuilt \(N/k\) times. Observed cadence matches the request. The exact field has no linear leftover vs \(1/k\). Report renderer accepts an AM-05 summary. |
| Does not prove | Prolongation/restriction quality, tagging (AM-03), MPI, performance, fitted spatial order. Native compile requires Kokkos + a compiler. |
| Resources | Local authoring and in-memory contract. Native path is optional. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compile records compiler/Kokkos only when `run_native` runs. |
