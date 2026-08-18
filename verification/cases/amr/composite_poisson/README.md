# AM-10 — composite Poisson (two-level residual)

Phase 3 code-verification case. Reuses the PO-01 trigonometric Poisson
oracle. In-memory two-level leaf residual remains the composite oracle.
Public authoring is a 1-d periodic AMR Case: stationary zero-flux
Program, `field_operator` with `FieldOutput("potential")` and
`GradientOutput(..., sign=-1)`, `ForwardEuler(..., fields=field)`,
`GeometricMG` + `CompositeHierarchySolve` (FFT is refused on AMR).
Two levels, ratio 2. `Tag(U > 0)` on `U = -ρ` covers the fine patch
`[0.5, 1]` (`exact.py` `INTERFACE=0.5`). `run_native` compiles, binds,
and advances one step, or raises `NativeUnavailable` without a
compiler/Kokkos. No private multilevel Poisson solver.

| Field | Content |
|---|---|
| Identifier | `AM-10` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | \(-\phi''=\rho\) on a 1-d periodic interval. Manufactured \(\phi=\sin(2\pi x)\), \(\rho=(2\pi)^2\phi\). Two levels: coarse \(N=16\) plus a fine patch on \([0.5,1]\) at ratio 2. |
| Oracle | PO-01 `phi_exact` / `rhs_exact` / `e_exact` loaded with `load_sibling_module` from `verification/cases/poisson/periodic_trig/exact.py`. `exact.py` does not read PoPS output. The composite leaf residual is the in-memory oracle (`two_level_residual`). |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\) (`CartesianDomain` + `Cartesian1D`). Fine patch covers the right half. Covered coarse cells are parents, not leaves. |
| Parameters | Dimensionless. Two levels. Injected covered-parent residual \(10^6\) is the negative control. |
| Native dimensions | `POPS_NATIVE_DIM=1`. Public AMR + composite GeometricMG may compile when the compiler/Kokkos is present. |
| Required capabilities | In-memory path: none. Native path: compiler + Kokkos (`missing_compiler_requirement` / `missing_native_compile_requirement`). |
| Configurations | Two-level leaf-only residual. Public 1-d AMR resolve. Optional native one-step field solve. No 3/4-level campaign, no MPI. |
| Diagnostics | Task 13 leaf-only L1/L2/L∞ of the composite residual. Full-grid L∞ sees the parent defect; leaf L2 does not. Native `run_native` returns a finite potential. |
| Thresholds | Leaf residual L2 / L∞ = 0 after excluding covered parents. |
| Proves | PO-01 reuse via `load_sibling_module`. Covered parent excluded from the two-level residual. Public AMR Case validates and resolves. Report renderer accepts an AM-10 summary. |
| Does not prove | Native composite FAC accuracy, 3–4 level MG, coarse-fine flux, rank invariance, 2-d/3-d, performance. |
| Resources | Local in-memory contract plus optional one-rank native compile. No two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native skip records compiler/Kokkos absence. |
