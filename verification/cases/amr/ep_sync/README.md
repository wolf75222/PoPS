# AM-11 — Euler–Poisson AMR sync

Phase 3 leaf-only charge contract plus public 1-d periodic Euler–Poisson
AMR authoring. Net charge on a two-level hierarchy is assembled from leaf
cells only. A covered parent must not be added a second time. The public
Case reuses CP-02 hydro + Poisson (`SSPRK2(..., fields=)`, operator
`fields`, `model.aux("potential")` / `model.aux("phi_grad_x")`) and the
AM-10 layout (Cartesian1D, two levels, ratio 2, frozen right-half patch,
`GeometricMG` + `CompositeHierarchySolve`, `EllipticRecompute` on field
transfer). FFT is refused on AMR. `run_native` compiles, binds, and
returns the level-0 electron conserved state `(2, n)`, or raises
`NativeUnavailable` without a compiler/Kokkos. No private EP AMR solver.

| Field | Content |
|---|---|
| Identifier | `AM-11` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | Euler–Poisson charge \(\rho_q=q n\). Two-level 1-d AMR: coarse leaf on \([0,0.5)\), covered parent on \([0.5,1)\) with two fine children. Net charge \(Q=\sum_{i\in\mathrm{leaf}} \rho_{q,i} V_i\). |
| Oracle | \(n=n_0+\delta\cos(2\pi x)\), \(q=1\), \(n_0=1\), \(\delta=0.25\). Parent density is the conservative restriction of the fine children. `exact.py` does not read PoPS output. |
| Domain and boundaries | Periodic unit interval \([0,1]\). Samples are cell centers of the two-level fixture. |
| Parameters | Dimensionless. Interface at \(x=0.5\). Four cells: one coarse leaf, one covered parent, two fine leaves. |
| Native dimensions | `POPS_NATIVE_DIM=1`. Public AMR + GeometricMG may compile when the compiler/Kokkos is present. |
| Required capabilities | In-memory path: none. Native path: compiler + Kokkos (`missing_compiler_requirement` / `missing_native_compile_requirement`). |
| Configurations | Leaf-only composition vs naive all-cell sum (double-count negative control). Public 1-d Euler–Poisson AMR resolve. Optional native `run_native(n_cells, t_end)`. |
| Diagnostics | Leaf net charge, covered-parent charge, naive all-cell sum, restricted fine-patch charge. |
| Thresholds | Leaf net charge equals the leaf subset sum. Covered parent \(\neq 0\) and equals the restricted children. Naive sum \(=\) leaf + parent and is not the composed charge. |
| Proves | Charge on leaves only. Covered parent is excluded even when its charge is large. Conservative restriction of the parent matches the fine children. Public Euler–Poisson AMR Case validates and resolves. Report renderer accepts an AM-11 summary with `amr.invariants_ok`. |
| Does not prove | Native composite FAC accuracy, subcycling field interpolation, reflux/re-solve after sync, Gauss-law residual on the live hierarchy, MPI, performance. |
| Resources | Local in-memory contract plus optional one-rank native compile. No two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native skip records compiler/Kokkos absence. |
