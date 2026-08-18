# PF-05 — composite AMR Poisson stand-in

Phase 0 kernel stand-in plus optional Phase 7 timing. Reuses the AM-10 /
PO-01 trigonometric Poisson oracle. In-memory two-level residual on leaf
cells only. Covered parents are excluded by Task 13
`leaf_reference_errors`. The refined fraction is an observation.
Optional `run_native` times AM-10
`verification/cases/amr/composite_poisson/run.py` when that helper is
present and returns `{elapsed_s, residual_or_error, cells_per_second}`.

| Field | Content |
|---|---|
| Identifier | `PF-05` |
| `verification_kind` | `infrastructure` |
| `evidence_status` | `required` |
| Equations | \(-\phi''=\rho\) on a 1-d periodic interval. Manufactured \(\phi=\sin(2\pi x)\), \(\rho=(2\pi)^2\phi\). Two levels: coarse \(N=16\) plus a fine patch on \([0.5,1]\) at ratio 2. |
| Oracle | AM-10 `phi_exact` / `rhs_exact` / `e_exact` (themselves PO-01) loaded with `load_sibling_module` from `verification/cases/amr/composite_poisson/exact.py`. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Fine patch covers the right half. Covered coarse cells are parents, not leaves. |
| Parameters | Dimensionless. Two levels. Injected covered-parent residual \(10^6\) is the negative control. Refined fraction is observed, not gated. |
| Native dimensions | `POPS_NATIVE_DIM=1`. Optional `run_native` reuses AM-10's 1-d AMR GeometricMG Case. |
| Required capabilities | None on the in-memory path. Optional `run_native` needs the AM-10 compiler/Kokkos path. |
| Configurations | Two-level leaf-only residual stand-in. Optional timed wrap of AM-10 `run_native`. No 3/4-level campaign, no MPI. |
| Diagnostics | Task 13 leaf-only L1/L2/L∞ of the composite residual. Full-grid L∞ sees the parent defect; leaf L2 does not. Fine-patch volume fraction is recorded as an observation. |
| Thresholds | Leaf residual L2 / L∞ = 0 after excluding covered parents. Refined fraction has no threshold. |
| Proves | AM-10 / PO-01 reuse via `load_sibling_module`. Leaf residual is defined. Covered parent is not double-counted. Report renderer accepts a PF-05 summary with empty `orders`. Optional `run_native` times AM-10 when the compiler is present. |
| Does not prove | Native composite FAC accuracy, 3–4 level MG, coarse-fine flux, rank invariance, 2-d/3-d, GPU. `elapsed_s` includes AM-10 compile+bind+solve. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
