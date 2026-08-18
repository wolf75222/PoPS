# AM-09 — AMR reflux conservation

Phase 3 AMR conservation contract. Documents that a closed two-level
statement is residual-zero only when the reflux term is present. Omitting
reflux is the open negative control. Residuals come from Task 16
`conservation_residual`. The in-memory ledger remains; a public 1-d
periodic AMR Case (AM-01 / TR-01 sine) validates and resolves on a live
two-level hierarchy so conservative reflux sits on the public AMR transfer
path. `run_native` compiles, binds, and runs when Kokkos and a compiler
are present.

| Field | Content |
|---|---|
| Identifier | `AM-09` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | Already-reduced discrete balance \(\mathrm{residual}=\mathrm{storage\_change}+\mathrm{outward\_boundary\_flux}-\mathrm{sources}-\mathrm{reflux}-\mathrm{projection}\). Periodic, no sources, no projection. Ratio-2 face: \(F_{\mathrm{fine}}=\tfrac12(f_0+f_1)\). Reflux \(=F_{\mathrm{fine}}-F_{\mathrm{coarse}}\). Live physics is TR-01 scalar sine advection \(a=1\). |
| Oracle | Manufactured coarse face flux \(1.0\), fine subfaces \((1.2, 1.6)\). Closed: leftover storage equals reflux. Open: same leftover, reflux omitted. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\) via `CartesianDomain` + `Cartesian1D`. Static two-level tag on \(x>0.5\). Frozen regrid. Ledger contract stays available without a mesh. |
| Parameters | Dimensionless. Ratio 2. Default coarse flux \(1.0\), fine fluxes \(1.2\) and \(1.6\). Default \(n_{\mathrm{coarse}}=8\). |
| Native dimensions | `POPS_NATIVE_DIM=1`. `run_native` returns the level-0 tracer as shape `(n_coarse,)`. |
| Required capabilities | Public AMR (`layouts.AMR`, `AMRHierarchy(max_levels=2, ratios=(2,))`), `AMRTransfer` + `StateTransfer`, default `FluxRegisterReflux`, Cartesian 1-d periodic, SSPRK2. Task 16 `conservation_residual` for the in-memory ledger. |
| Configurations | Closed with reflux vs open without (in-memory). Frozen spatial marker on the right half. No invented user-facing reflux-off switch on `run_native`. |
| Diagnostics | Task 16 `conservation_residual`. Closed residual 0. Open residual nonzero (negative control). |
| Thresholds | Closed residual \(=0\). Open residual \(\ne 0\). No fitted-order gate. |
| Proves | Closed reflux statement has residual 0. Open statement without reflux is a detectable leftover. Public 1-d periodic AMR Case validates and resolves with reflux on the transfer path. Report renderer accepts an AM-09 summary. |
| Does not prove | Native flux-register conservation to round-off, MPI gather, multi-patch coverage, average_down-only, subcycling order, interface-band error, performance. Native compile requires Kokkos + a compiler. |
| Resources | Local authoring and in-memory ledger. Native path is optional. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compile records compiler/Kokkos only when `run_native` runs. |
