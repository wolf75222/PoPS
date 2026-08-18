# IF-04 — checkpoint / restart identity

Infrastructure identity. Serialize a TR-01 state dict (centers, q, t)
to JSON and reload. Array identity after the round-trip. A Case can
install `pops.output.consumers.Checkpoint`. `run_native` then raises
`NativeUnavailable`: `restore_checkpoint_payload` needs owner+executor
and there is no public path restore that returns `{centers, q, t}`.

| Field | Content |
|---|---|
| Identifier | `IF-04` |
| `verification_kind` | `infrastructure` |
| `evidence_status` | `required` |
| Equations | \(\partial_t q + a\partial_x q = 0\). Conservative scalar \(q\). Same IC as TR-01: \(q=q_0+\varepsilon\sin(2\pi k(x-at))\). |
| Oracle | TR-01 checkpoint `{centers, q, t}` via `load_sibling_module` on `advection_sine/exact.py`. JSON dump/load must recover the arrays with \(L^\infty=0\) and preserve \(t\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Samples are interior cell centers. |
| Parameters | \(n=32\) uniform cells. \(t=0.25\). TR-01 defaults \(q_0=1\), \(\varepsilon=10^{-2}\), \(k=1\), \(a=1\). |
| Native dimensions | `POPS_NATIVE_DIM=1` listed for the planner. Optional `run_native` installs `Checkpoint` then refuses restore. |
| Required capabilities | None (`requires = []`). KokkosSerial listed for the planner; MPI off. JSON stays the in-memory oracle. |
| Configurations | Single manufactured TR-01 state. Native restore is refused; no invented executor API. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ of restored vs original `centers` and `q`. Equality of `t`. |
| Thresholds | Round-trip \(L^\infty=0\). `t` preserved. Empty `orders` with reason `in-memory checkpoint identity`. |
| Proves | JSON serialization of a TR-01 checkpoint dict is an identity on arrays and time. `Checkpoint` can be installed on the TR-01 Case. Report renderer accepts an IF-04 summary. |
| Does not prove | Live `RuntimeInstance.checkpoint(path)` write, `restore_checkpoint_payload` apply, MPI rank remapping, AMR rematerialization, bit-identical native restart, spatial/temporal order. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
