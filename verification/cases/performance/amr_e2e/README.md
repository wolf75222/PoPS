# PF-11 — dynamic AMR e2e

Phase 0 performance stand-in. Two warmup steps, then 50 fake steps with a
regrid every 8. Measured rebuilds are \(50/8\). Warmup rebuilds are recorded
and dropped. Leaf-cell throughput is cells / fake time. In-memory only; no
live runtime, compile, or timed PF run.

| Field | Content |
|---|---|
| Identifier | `PF-11` |
| `verification_kind` | `infrastructure` |
| `evidence_status` | `required` |
| Equations | No PDE. Fake step loop with a prescribed regrid cadence. Two-level 1-d leaf count: uncovered coarse plus refined half at ratio 2. |
| Oracle | Rebuilds on global steps \(0,8,16,\ldots\). Warmup prefix is 2 steps, so step 0 is dropped. Measured rebuilds \(=50/8=6\). Throughput \(=\) (leaf cells \(\times\) 50) / fake time. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Coarse \(N=16\), interface at \(x=0.5\), refinement ratio 2. Leaf cells only. |
| Parameters | Dimensionless. Warmup 2. Measured steps 50. Regrid every 8. Fake step time \(10^{-3}\). |
| Native dimensions | `POPS_NATIVE_DIM=1` listed for the planner. Optional `run_native` wraps a short AM-01 or AM-02 path. |
| Required capabilities | None on the in-memory path. Optional `run_native` times public AM-01 (default) or AM-02. |
| Configurations | Single 1-d stand-in. No resolution series, no live hierarchy, no timed PF run in this increment. |
| Diagnostics | Measured rebuild count. Warmup rebuild count (excluded). Leaf-cell throughput observation. |
| Thresholds | Rebuilds \(=50/8=6\). Warmup steps do not appear in the measured rebuild list. Throughput \(>0\) (observation, not a cells/s gate). |
| Proves | The fake e2e loop rebuilds every 8 steps. The 2-step warmup is not counted. Report renderer accepts a PF-11 summary with a fake one-node throughput observation and empty `orders`. Optional `run_native` wraps AM-01 / AM-02. |
| Does not prove | Native dynamic AMR, tagging, clustering, prolongation/restriction, MPI regrid, GPU, two-node scaling, or a timed cells/s claim. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
