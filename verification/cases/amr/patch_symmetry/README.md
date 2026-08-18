# AM-12 — patch-shape / reflection symmetry

Phase 3 leftover AMR contract plus the live reflected counterpart of AM-01.
Reuses the TR-01 manufactured sine on a static 1-d two-level coarse-fine
partition and its image under \(x\mapsto 1-x\). Interface-band errors of a
leftover that is itself reflected must match. The public Case tags the left
half \(x<0.5\) so a frozen fine patch covers \([0, 0.5]\). Leftover in-memory
helpers remain; `run_native` compiles, binds, and runs when Kokkos and a
compiler are present.

| Field | Content |
|---|---|
| Identifier | `AM-12` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | \(\partial_t q + a \partial_x q = 0\). Conservative scalar \(q\). Canonical \(a=1\). Static two-level AMR intent: fine left of \(x=0.5\), coarse right. |
| Oracle | TR-01 sine via `load_sibling_module` on `verification/cases/transport/advection_sine/exact.py`. Interface remains at \(x=0.5\) after \(x\mapsto 1-x\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\) via `CartesianDomain` + `Cartesian1D`. Interface \(\Gamma_{cf}\) at \(x=0.5\). Original leftover mesh: coarse left, fine right. Reflected live Case: fine left, coarse right. Distance \(d(x,\Gamma_{cf})=\lvert x-0.5\rvert\). |
| Parameters | Dimensionless. Default \(n_{\mathrm{coarse}}=8\), refinement ratio 2, band \(m=4\) fine cells (plan §9.5). |
| Native dimensions | `POPS_NATIVE_DIM=1`. `run_native` returns the level-0 tracer as shape `(n_coarse,)`. |
| Required capabilities | Public AMR (`layouts.AMR`, `AMRHierarchy(max_levels=2, ratios=(2,))`), Cartesian 1-d periodic, SSPRK2. Task 19 `interface_band_mask` / `band_max_abs_error` for leftover observations. |
| Configurations | Frozen regrid after a spatial marker tag \(x<0.5\). Manufactured leftover is reflected with the oracle. No resolution series in this increment. |
| Diagnostics | \(E_{cf}=\max_{d<m h_f}\lvert U_h-U^{\mathrm{exact}}\rvert\) on the original band and on the reflected leftover+oracle. Independent reflected mask vs spatially reversed original mask. |
| Thresholds | Reflected mask equals the spatially reversed original mask. \(E_{cf}\) is unchanged under reflection of leftover+oracle. Exact-vs-exact \(E_{cf}=0\). |
| Proves | Public 1-d periodic AMR Case validates and resolves with the reflected interface. The reflected interface band is the spatial reverse of the original band. Leftover interface-band error is invariant under \(x\mapsto 1-x\). Report renderer accepts an AM-12 summary with measured interface/bulk errors and `amr.invariants_ok`. |
| Does not prove | Live regrid motion, reflux, subcycling order retention, MPI, performance. Native compile requires Kokkos + a compiler. |
| Resources | Local authoring and leftover contract. Native path is optional. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compile records compiler/Kokkos only when `run_native` runs. |
