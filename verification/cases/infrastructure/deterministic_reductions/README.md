# IF-06 — deterministic reductions

Phase 1 infrastructure contract. Sum a 1-d float64 field with sequential,
pairwise, and blocked reductions. For the exactly representable geometric
series \(2^{-k}\), all three match the closed form bitwise. A chaotic leftover
mix of magnitudes may differ (observation only). In-memory sequential /
pairwise / blocked sums stay the oracle. Phase 6 `run_native` refuses a
native reduction: `refuse_deterministic_reduction_switch()` returns
`public deterministic-reduction switch not active`. No public switch
selects a compiled all-reduce tree.

| Field | Content |
|---|---|
| Identifier | `IF-06` |
| `verification_kind` | `infrastructure` |
| `evidence_status` | `required` |
| Equations | No PDE. Scalar reduction identity of a 1-d field: sequential accumulation, pairwise binary tree, and blocked (width 8) sequential-of-sequentials. |
| Oracle | Geometric series \(q_k=2^{-k}\) for \(k=0,\ldots,n-1\). Closed form \(Q=2-2^{1-n}\). All terms and the sum are exact in float64. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d cell vector of length \(n=32\). No spatial operator, no boundary flux. |
| Parameters | Dimensionless. Default \(n=32\). Block width 8. Chaotic leftover is a tiled mix of \(\{10^{16},1,-10^{16},10^{-8},-1,10^{-8}\}\) (observation only). |
| Native dimensions | `POPS_NATIVE_DIM=1` listed for the planner. `run_native` does not load a native artifact. |
| Required capabilities | None on the in-memory path. Native trees require `deterministic_reduction_switch` (currently false). |
| Configurations | One exact-representable series and one chaotic leftover. Physics, resolution, and \(t\) are unused. Only the reduction tree changes. |
| Diagnostics | Bitwise identity of the three exact-representable sums against the closed form. Chaotic leftover sums are recorded, not gated. |
| Thresholds | Exact-representable sums agree bitwise. No spatial-order gate on the in-memory path. |
| Proves | Sequential, pairwise, and blocked reductions of an exactly representable float64 geometric series are identical. Report renderer accepts an IF-06 summary with empty `orders` and reason `reduction identity / no live MPI`. `run_native` names the missing switch. |
| Does not prove | Live MPI all-reduce, a public sequential/pairwise/blocked switch, rank-count invariance of a compiled Program, Kahan / compensated summation, thread/GPU invariance, spatial order, AMR, Poisson, coupling, performance. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
