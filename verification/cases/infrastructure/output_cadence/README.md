# IF-05 — output-cadence invariance

Phase 0 infrastructure contract. Reuses the TR-01 manufactured sine and
advances it analytically to \(t=0.25\). Dump every \(1/2/10\) steps by
evaluating the same exact state. Dumps copy the field and do not mutate
it. Field-to-field \(L^\infty\) is 0 across cadences. In-memory oracle
stays green. Phase 6 `run_native` refuses two TR-01 dump cadences:
`refuse_output_cadence_consumer()` returns
`public output-cadence consumer attach not active`. A public
`ScientificOutput` / `ConsumerGraph` exists, but TR-01 `build_case` does
not expose a clock or state Handle, so a dump-every-step consumer cannot
be attached without changing the Case.

| Field | Content |
|---|---|
| Identifier | `IF-05` |
| `verification_kind` | `infrastructure` |
| `evidence_status` | `required` |
| Equations | \(\partial_t q + a\partial_x q = 0\). Conservative scalar \(q\). Canonical \(a=1\). Same IC as TR-01: \(q=q_0+\varepsilon\sin(2\pi k x)\). |
| Oracle | Reused TR-01 manufactured sine via `load_sibling_module` on `verification/cases/transport/advection_sine/exact.py`. Sampled independently at each dump time of cadences \(1\), \(2\), and \(10\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Samples are interior cell centers. |
| Parameters | Dimensionless. Default in-memory grid \(N=32\). \(N_{\mathrm{steps}}=10\), \(\Delta t=0.025\), \(T=0.25\). Cadences \(k\in\{1,2,10\}\). TR-01 defaults \(q_0=1\), \(\varepsilon=10^{-2}\), \(k=1\), \(a=1\). Leftover mutation amplitude \(1/4\) (dyadic). |
| Native dimensions | `POPS_NATIVE_DIM=1` listed for the planner. `run_native` does not load a native artifact. |
| Required capabilities | None on the in-memory path. Native dump cadences require `output_cadence_consumer` (currently false). |
| Configurations | Three dump cadences. Physics, resolution, and \(t=0.25\) stay fixed. Only the dump interval changes. |
| Diagnostics | Pairwise field-to-field L1/L2/L∞ between the three cadence finals. Dump count \(N/k\). L∞ leftover of a one-cell mutated dump versus the unmutated field. |
| Thresholds | Difference between cadences is \(L^\infty=0\). Mutated leftover dump \(L^\infty\) equals the mutation amplitude. No spatial-order gate on the in-memory path. |
| Proves | Exact fields are independent of dump cadence. Dumps copy the field. A mutated leftover dump is detected. Report renderer accepts an IF-05 summary with empty `orders` and reason `exact-field identity / no live output cadence`. `run_native` names the missing attach. |
| Does not prove | Live HDF5 / NPZ / ParaView output, ConsumerGraph `every(k)` on a compiled Program, two native TR-01 runs with and without dumps, MPI, thread/GPU invariance, spatial order, AMR, Poisson, coupling, performance. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
