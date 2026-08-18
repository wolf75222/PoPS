# AM-01 — static coarse-fine wave

Phase 3 AMR interface-band contract. Reuses the TR-01 manufactured sine on a
1-d periodic unit interval with a public two-level AMR hierarchy. The right
half \(x>0.5\) is tagged so a fine patch covers \([0.5, 1]\). Leftover
in-memory helpers remain; `run_native` compiles, binds, and runs when Kokkos
and a compiler are present.

| Field | Content |
|---|---|
| Identifier | `AM-01` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | \(\partial_t q + a \partial_x q = 0\). Conservative scalar \(q\). Canonical \(a=1\). Static two-level AMR intent: coarse left of \(x=0.5\), fine right. |
| Oracle | TR-01 sine via `load_sibling_module` on `verification/cases/transport/advection_sine/exact.py`. Translation \(q(x,t)=q(x-at,0)\) still holds on the leaf centers. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\) via `CartesianDomain` + `Cartesian1D`. Interface \(\Gamma_{cf}\) at \(x=0.5\). Distance \(d(x,\Gamma_{cf})=\lvert x-0.5\rvert\). |
| Parameters | Dimensionless. Default \(n_{\mathrm{coarse}}=8\), refinement ratio 2, band \(m=4\) fine cells (plan §9.5). |
| Native dimensions | `POPS_NATIVE_DIM=1`. `run_native` returns the level-0 tracer as shape `(n_coarse,)`. |
| Required capabilities | Public AMR (`layouts.AMR`, `AMRHierarchy(max_levels=2, ratios=(2,))`), Cartesian 1-d periodic, SSPRK2. Task 19 `interface_band_mask` / `band_max_abs_error` for leftover observations. |
| Configurations | Frozen regrid after a spatial marker tag \(x>0.5\). Manufactured leftover is an observation, not a pass/fail gate. No resolution series in this increment. |
| Diagnostics | \(E_{cf}=\max_{d<m h_f}\lvert U_h-U^{\mathrm{exact}}\rvert\). \(E_{\mathrm{bulk}}\) on the complement. Ratio \(E_{cf}/E_{\mathrm{bulk}}\) is recorded. |
| Thresholds | \(E_{cf}\) is defined and finite. Exact-vs-exact \(E_{cf}=0\). Leftover \(E_{cf}/E_{\mathrm{bulk}}\) is an observation, not a raise. |
| Proves | Public 1-d periodic AMR Case validates and resolves. TR-01 translation identity on the static CF mesh. Task 19 band helpers apply with \(d=\lvert x-0.5\rvert\). Report renderer accepts an AM-01 summary with measured interface/bulk errors. |
| Does not prove | Live regrid motion, reflux, subcycling order retention, MPI, performance. Native compile requires Kokkos + a compiler. |
| Resources | Local authoring and leftover contract. Native path is optional. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compile records compiler/Kokkos only when `run_native` runs. |
