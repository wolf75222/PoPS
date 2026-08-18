# IF-07 — native / DSL / hybrid parity

Phase 0 infrastructure contract. Reuses the TR-01 manufactured sine and
samples it independently under the three path labels native, dsl, and
hybrid. Field-to-field \(L^\infty\) is 0. The public TR-01 Case
(`public_dsl_case`) **is** the DSL path. Phase 6 does not invent a
second authoring stack: `refuse_hybrid_native_cpp()` returns
`public hybrid/native C++ authoring not active`, and `run_native`
raises `NativeUnavailable` with that string. Live compile of the public
Case remains TR-01 `run_native`.

| Field | Content |
|---|---|
| Identifier | `IF-07` |
| `verification_kind` | `infrastructure` |
| `evidence_status` | `required` |
| Equations | \(\partial_t q + a\partial_x q = 0\). Conservative scalar \(q\). Canonical \(a=1\). Same IC as TR-01: \(q=q_0+\varepsilon\sin(2\pi k x)\). |
| Oracle | Reused TR-01 manufactured sine via `load_sibling_module` on `verification/cases/transport/advection_sine/exact.py`. Sampled independently under each path label. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Path labels: `native`, `dsl`, `hybrid`. |
| Parameters | Dimensionless. Default in-memory grid \(N=32\). TR-01 defaults \(q_0=1\), \(\varepsilon=10^{-2}\), \(k=1\), \(a=1\). |
| Native dimensions | `POPS_NATIVE_DIM=1` listed for the planner. IF-07 `run_native` does not load a native artifact. |
| Required capabilities | None on the in-memory path. Hybrid / native C++ authoring requires `hybrid_native_cpp_authoring` (currently false). The public Case is the DSL path. |
| Configurations | Three in-memory path labels. Physics, resolution, and \(t\) stay fixed. Only the authoring label changes. DSL authoring is TR-01 `build_case`. |
| Diagnostics | Pairwise field-to-field L1/L2/L∞ between the three exact path labels. |
| Thresholds | Difference between paths is \(L^\infty=0\). No spatial-order gate on the in-memory path. |
| Proves | Exact fields are independent of the native / DSL / hybrid label. The public Case is the DSL path. Report renderer accepts an IF-07 summary with empty `orders` and reason `exact-field identity / no live native-DSL-hybrid`. `run_native` names the missing hybrid / native C++ stack. |
| Does not prove | A second native C++ authoring stack, hybrid backend, live compile inside IF-07, spatial order, AMR, Poisson, coupling, performance. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
