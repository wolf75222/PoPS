# AM-07 — AMR vs uniform fine

Phase 3 AMR comparison. Three series: uniform \(h\), AMR base \(h\) with a
local fine patch at \(h/2\), and uniform \(h/2\). Public Cases author each
series (`build_case` / `resolve_plan`). Fine-region AMR error must match
uniform \(h/2\) under the manufactured \(E\propto h^2\) model. In-memory
helpers remain; `run_native` compiles, binds, and runs when Kokkos and a
compiler are present.

| Field | Content |
|---|---|
| Identifier | `AM-07` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | Manufactured second-order spatial error \(E=C h^2\). Public series integrate \(\partial_t q + a \partial_x q = 0\) on the unit interval. The comparison is local: the refined patch uses \(h_f=h/2\). |
| Oracle | `manufactured_error(h)=C h^2` with \(C=1\). `FINE_RATIO=2`. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\) via `CartesianDomain` + `Cartesian1D`. Canonical base \(h=1/16\). AMR interface at \(x=0.5\). |
| Parameters | Dimensionless. Series `uniform_h` (\(n\) cells), `amr_h_fine_h2` (\(n\) coarse cells, fine right half), `uniform_h2` (\(2n\) cells). Default \(n=16\). |
| Native dimensions | `POPS_NATIVE_DIM=1`. `run_native` returns a 1-d tracer: uniform full field, or AMR level-0 of shape `(n,)`. |
| Required capabilities | Public Uniform (`uniform_periodic_layout`) and public AMR (`layouts.AMR`, `AMRHierarchy(max_levels=2, ratios=(2,))`), Cartesian 1-d periodic, SSPRK2. |
| Configurations | Uniform \(h\); AMR \(h\) + fine \(h/2\); uniform \(h/2\). Frozen regrid after a spatial marker tag \(x>0.5\). No reflux in this increment. |
| Diagnostics | Fine-region AMR error vs uniform \(h/2\). Task 15 observed spatial order on the manufactured \(E\propto h^2\) series. |
| Thresholds | Fine-region AMR error equals uniform \(h/2\). Observed order \(=2\) (gate \(\ge 1.8\)). Uniform-\(h\) error is four times the fine-region error. |
| Proves | Three-series local-spacing contract. Public Cases validate and resolve for all three series names. Fine-region AMR matches uniform \(h/2\). Manufactured second-order spatial error. Report renderer accepts an AM-07 summary with `amr.order_retained=true`. |
| Does not prove | Live fine-region vs uniform-\(h/2\) native order, interface-band layer (AM-01/AM-08), reflux (AM-09), subcycling (AM-04), MPI, performance. Native compile requires Kokkos + a compiler. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
