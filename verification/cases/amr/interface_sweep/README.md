# AM-08 — interface placement sweep

Phase 3 AMR interface-placement contract. Sweeps the coarse-fine interface
\(\Gamma_{cf}\) at \(x_0\in[0,1)\). In-memory manufactured second-order
error \(E\propto h^2\) remains. A public 1-d periodic AMR Case authors a
static two-level hierarchy whose marker is parameterized by \(x_0\)
(`where(x > x0, 1, 0)` wrapping periodically). Frozen regrid. `run_native`
compiles, binds, and runs when Kokkos and a compiler are present.

| Field | Content |
|---|---|
| Identifier | `AM-08` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | 1-d periodic advection of the TR-01 sine. Static two-level interface at parameterized \(x_0\). Manufactured discrete error \(E(x_0)=C(x_0)\,h^2\). |
| Oracle | TR-01 `exact_sine` loaded with `load_sibling_module`. Placement scale \(C(x_0)=1+\tfrac12\sin(2\pi x_0)\). Interface-band \(E_{cf}\) via Task 19 `interface_band_mask` / `band_max_abs_error`. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\) via `CartesianDomain` + `Cartesian1D`. Interface placements \(x_0=i/8\), \(i=0,\ldots,7\). Distance is the periodic \(d(x,\Gamma_{cf})\). |
| Parameters | Dimensionless. Default \(n_{\mathrm{cells}}=32\), default \(x_0=0.5\), refinement ratio 2, band \(m=4\) fine cells. Scale \(C_0=0.04\). |
| Native dimensions | `POPS_NATIVE_DIM=1`. `run_native` returns the level-0 tracer as shape `(n_cells,)`. |
| Required capabilities | Public AMR (`layouts.AMR`, `AMRHierarchy(max_levels=2, ratios=(2,))`), Cartesian 1-d periodic, SSPRK2. Task 19 `interface_band_mask` / `band_max_abs_error` for leftover observations. |
| Configurations | Frozen regrid after a spatial marker tag on the periodic half-interval starting at \(x_0\). Uniform manufactured \(E\propto h^2\) on each placement. No subcycling, no live regrid. |
| Diagnostics | Error envelope \((\min E_{cf},\max E_{cf})\). Worst-placement observed order of \(E\propto h^2\). Task 19 \(E_{cf}\) / \(E_{\mathrm{bulk}}\). |
| Thresholds | Envelope entries finite and positive. Worst-placement observed order \(\approx 2\) (threshold \(1.8\)). |
| Proves | Public 1-d periodic AMR Case validates and resolves at a parameterized \(x_0\). Envelope over \(x_0\in[0,1)\) is finite. The worst placement still retains manufactured spatial order 2. Report renderer accepts an AM-08 summary. |
| Does not prove | Live regrid motion, native transport order, reflux, subcycling, moving patches, MPI, performance. Native compile requires Kokkos + a compiler. |
| Resources | Local authoring and in-memory sweep. Native path is optional. `pr.nodes = 1`. `two_node.nodes = [1, 2]`. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compile records compiler/Kokkos only when `run_native` runs. |
