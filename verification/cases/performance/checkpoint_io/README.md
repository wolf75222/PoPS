# PF-10 — checkpoint / HDF5 stand-in

Phase 1 kernel stand-in. Write/read a numpy npz of a unique 1-d field.
Record bytes and a canned write-time observation. A no-output path must
not create the artifact. No live HDF5, compile, or timed PF run.

| Field | Content |
|---|---|
| Identifier | `PF-10` |
| `verification_kind` | `infrastructure` |
| `evidence_status` | `required` |
| Equations | No PDE. Identity of a 1-d field through `numpy.savez` / `numpy.load`. |
| Oracle | Unique interior pattern \(q_i=i+1\). Reload must recover the array with \(L^\infty=0\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Samples are \(N=32\) cell values. |
| Parameters | Dimensionless. Default \(N=32\). Fake write-time \(10^{-3}\,\mathrm{s}\). Artifact name `checkpoint.npz`. |
| Native dimensions | `POPS_NATIVE_DIM=1` listed for the planner. Optional `run_native` wraps IF-04, else TR-01. |
| Required capabilities | None on the in-memory path. Optional `run_native` times IF-04 checkpoint/restart or TR-01. |
| Configurations | Single 1-d stand-in. Output path writes npz. No-output path writes nothing. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ of restored vs original \(q\). Recorded file bytes and fake write-time. |
| Thresholds | Round-trip \(L^\infty=0\). Output bytes \(>0\). No-output path creates no artifact. Empty `orders` with reason `npz checkpoint stand-in, not a timed PF run`. |
| Proves | npz dump/load is an identity on the 1-d field. The no-output path performs no implicit I/O. Report renderer accepts a PF-10 summary with empty `orders`. Optional `run_native` wraps IF-04 / TR-01. |
| Does not prove | Live HDF5, MPI collective I/O, AMR rematerialization, bit-identical native restart, timed GB/s, spatial/temporal order. This is not a timed PF run. |
| Resources | Local npz contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
