# PF-07 — regrid / clustering stand-in

Phase 1 kernel-microbench stand-in. Tag the TR-02 1-d Gaussian pulse by
amplitude, then cluster contiguous tagged runs into patches of minimum width 4.
Count clustered patches against the raw tag count. In-memory only; no live
runtime, compile, or timed PF run.

| Field | Content |
|---|---|
| Identifier | `PF-07` |
| `verification_kind` | `infrastructure` |
| `evidence_status` | `required` |
| Equations | No PDE. Tag \(q>q_{\mathrm{tag}}\) on the TR-02 exact pulse. Each contiguous tagged run becomes one patch; runs shorter than 4 cells grow to width 4 (periodic). |
| Oracle | TR-02 translated Gaussian, loaded with `load_sibling_module`. Documented \(q_{\mathrm{tag}}=0.1\) on \(n=128\). Clustering oracle is the periodic contiguous-run cover. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Uniform cell centers. Patch growth is periodic. |
| Parameters | Dimensionless. TR-02 defaults \(x_0=0.37\), \(\sigma=0.08\), \(a=1\). Default \(N=128\). Tag threshold \(0.1\). Minimum patch width 4. |
| Native dimensions | `POPS_NATIVE_DIM=1` listed for the planner. Optional `run_native` wraps AM-02 / AM-03 / AM-05. |
| Required capabilities | None on the in-memory path. Optional `run_native` times public AM-02 (default), AM-03, or AM-05. |
| Configurations | Single 1-d stand-in. No live regrid, no resolution series, no timed PF run in this increment. |
| Diagnostics | Raw tag count versus clustered patch count. Boolean cover of tagged cells by the clustered patches. |
| Thresholds | Clustered patch count \(\le\) raw tag count. Every tagged cell is covered. Every patch has width \(\ge 4\). |
| Proves | Amplitude tagging of the TR-02 pulse is non-empty. Contiguous-run clustering with min width 4 covers every tagged cell and never emits more patches than raw tags. Report renderer accepts a PF-07 summary with empty `orders`. Optional `run_native` wraps AM-02 / AM-03 / AM-05. |
| Does not prove | Native Berger–Rigoutsos, MPI tag gather, live AMR regrid, proper nesting, timed cells/s, Poisson, coupling. This is not a timed PF run. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
