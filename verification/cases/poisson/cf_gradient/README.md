# PO-06 — CF gradient placement (1-d PO-01 oracle)

Phase 3 code-verification case. In-memory manufactured 1-d trigonometric
Poisson with a coarse-fine interface placed on extrema of the PO-01 field.
Public authoring is a 1-d periodic AMR Case: stationary zero-flux Program,
`field_operator` with `FieldOutput("potential")` and
`GradientOutput(..., sign=-1)`, `ForwardEuler(..., fields=field)`,
`GeometricMG` + `CompositeHierarchySolve` (FFT is refused on AMR).
Two levels, ratio 2, frozen CF interface at `x = 0.5`. `Tag(U > 0)` on
`U = -ρ` covers the fine patch `[0.5, 1]`. `run_native` compiles, binds
the manufactured RHS, and advances one step, or raises `NativeUnavailable`
without a compiler/Kokkos. In-memory placement helpers stay the CF-band
oracle. No private elliptic solver.

| Field | Content |
|---|---|
| Identifier | `PO-06` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | \(-\phi''=\rho\) on a 1-d periodic interval. Manufactured \(\phi=\sin(2\pi x)\), \(\rho=(2\pi)^2\phi\), \(E=-d\phi/dx=-2\pi\cos(2\pi x)\). |
| Oracle | Reuses the PO-01 1-d \(\phi\). `phi_exact` / `rhs_exact` / `e_exact` / `dphi_exact` / `d2phi_exact` plus named CF placements in `exact.py`. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Samples are uniform cell centers. Interface \(\Gamma_{cf}\) at 0.25 (max \(\phi\), max \(\lvert\phi''\rvert\)) or 0 (zero of \(\phi\), max \(\lvert\phi'\rvert\)). |
| Parameters | Dimensionless. Manufactured order series \(N=16,32,64,128\). Optional \(C h^2\) perturbation of \(E\), larger inside the four-cell CF band. |
| Native dimensions | `POPS_NATIVE_DIM=1`. Public AMR + composite GeometricMG may compile when the compiler/Kokkos is present. |
| Required capabilities | In-memory path: none. Native path: compiler + Kokkos (`missing_compiler_requirement` / `missing_native_compile_requirement`). |
| Configurations | In-memory placements remain uniform cell-center distances. Public 1-d AMR resolve: two-level frozen CF at 0.5, `GeometricMG` + `CompositeHierarchySolve`, `Periodic`, `ConstantNullspace`, `MeanValueGauge(0)`. Optional native one-step field solve. |
| Diagnostics | Task 19 \(E_{cf}\) / \(E_{bulk}\) at each placement. Task 15 observed order of manufactured \(E\propto h^2\). Residual of \(\rho-(2\pi)^2\phi\). Native `run_native` returns a finite potential. |
| Thresholds | Spatial order threshold \(1.8\) for a declared second-order field. No placement may drop \(E\) to order one. |
| Proves | The four CF locations match max \(\phi\), zero, max \(\lvert\phi'\rvert\), max \(\lvert\phi''\rvert\); manufactured field order 2 at every placement; public AMR Case validates and resolves; schema-valid campaign report. |
| Does not prove | Native composite FAC accuracy, live gradient reconstruction at every placement, 2-d/3-d, coupling, MPI, performance. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. Suites: nightly, weekly, release. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
