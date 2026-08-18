# AM-03 — gradient / second-diff tagging

Phase 3 AMR tagging contract. Reuses the TR-02 manufactured Gaussian on a
1-d periodic unit interval. In-memory helpers tag a cell when
\(\lvert\Delta q\rvert>\theta\) or the undivided second difference exceeds
\(\theta_2\). A public two-level AMR Case tags the live tracer
(`ValueExpr(tracer) > threshold`, coarsen below a lower threshold, buffer 2,
hysteresis). Leftover in-memory helpers remain; `run_native` compiles, binds,
and runs when Kokkos and a compiler are present.

| Field | Content |
|---|---|
| Identifier | `AM-03` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | Scalar advection \(\partial_t q + a\partial_x q = 0\). In-memory tagging is evaluated on the exact field. Undivided \(\lvert\Delta q\rvert_i=\max(\lvert q_i-q_{i-1}\rvert,\lvert q_{i+1}-q_i\rvert)\). Second difference \(\lvert q_{i+1}-2q_i+q_{i-1}\rvert\). The live Case tags the numerical tracer, not a prescribed marker. |
| Oracle | TR-02 translated Gaussian, loaded with `load_sibling_module`. Documented \(\theta=0.02\), \(\theta_2=0.002\) on \(n=128\). Pulse core: cells within \(0.5\sigma\) of the exact peak. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\) via `CartesianDomain` + `Cartesian1D`. Buffer dilation is periodic. |
| Parameters | TR-02 defaults \(x_0=0.37\), \(\sigma=0.08\), \(a=1\). \(\theta=0.02\), \(\theta_2=0.002\). Buffer widths \(1,2,4\) in-memory; live `Buffer(cells=2)`. Refine if above \(\theta\) (or \(\theta_2\)); coarsen if both indicators are below half those thresholds. Live regrid every 2 accepted steps. |
| Native dimensions | `POPS_NATIVE_DIM=1`. `run_native` returns the level-0 tracer as shape `(n_cells,)`. |
| Required capabilities | Public AMR (`layouts.AMR`, `AMRHierarchy(max_levels=2, ratios=(2,))`), Cartesian 1-d periodic, SSPRK2 on the tracer, live `AMRRegrid(schedule=every(N))`, `Tag` + `Coarsen` + `Buffer` + `Hysteresis`. |
| Configurations | Uniform-mesh in-memory contract plus live tracer tagging. No prescribed-patch comparison (AM-02) or fitted order series in this increment. |
| Diagnostics | Raw tag mask vs pulse core; periodic buffer-2 halo (exactly two cells on each side of a single run); hysteresis stationarity on the static field. |
| Thresholds | Tagged set contains the pulse core. Buffer of 2 adds exactly 4 halo cells (2 per side). Repeated hysteresis on a static field is identical. |
| Proves | Public 1-d periodic AMR Case validates and resolves. Gradient / second-diff tagging of the TR-02 pulse covers the documented core. Periodic dilation by 2 is a two-cell halo. Refine/coarsen hysteresis does not oscillate on a static field. Report renderer accepts an AM-03 summary. |
| Does not prove | Clustering, prescribed-patch comparison (AM-02), order retention, MPI, performance. Native compile requires Kokkos + a compiler. |
| Resources | Local authoring and in-memory contract. Native path is optional. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compile records compiler/Kokkos only when `run_native` runs. |
