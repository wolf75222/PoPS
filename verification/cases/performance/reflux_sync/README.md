# PF-08 — reflux / AMR sync stand-in

Phase 1 kernel-microbench stand-in. Reuses the AM-09 already-reduced
two-level reflux ledger: a closed statement is residual-zero only when
the reflux term is present. Omitting reflux is the open negative control.
Elapsed wall time of the residual evaluation is an observation, not a
threshold. Residuals come from Task 16 `conservation_residual`. In-memory
only; no live runtime, compile, or timed PF run.

| Field | Content |
|---|---|
| Identifier | `PF-08` |
| `verification_kind` | `infrastructure` |
| `evidence_status` | `required` |
| Equations | Already-reduced discrete balance \(\mathrm{residual}=\mathrm{storage\_change}+\mathrm{outward\_boundary\_flux}-\mathrm{sources}-\mathrm{reflux}-\mathrm{projection}\). Periodic, no sources, no projection. Ratio-2 face: \(F_{\mathrm{fine}}=\tfrac12(f_0+f_1)\). Reflux \(=F_{\mathrm{fine}}-F_{\mathrm{coarse}}\). |
| Oracle | Manufactured coarse face flux \(1.0\), fine subfaces \((1.2, 1.6)\). Closed: leftover storage equals reflux. Open: same leftover, reflux omitted. `exact.py` does not read PoPS output. |
| Domain and boundaries | Contract is ledger-local. Periodic unit interval implied so domain-boundary flux is 0. No spatial mesh is required for the in-memory path. |
| Parameters | Dimensionless. Ratio 2. Default coarse flux \(1.0\), fine fluxes \(1.2\) and \(1.6\). |
| Native dimensions | `POPS_NATIVE_DIM=1` listed for the planner. Optional `run_native` wraps AM-09. |
| Required capabilities | None on the in-memory path. Optional `run_native` times public AM-09 reflux. |
| Configurations | Closed with reflux vs open without. Elapsed wall time recorded as an observation. No spatial-order, subcycling, or timed PF campaign in this increment. |
| Diagnostics | Task 16 `conservation_residual`. Closed residual 0. Open residual nonzero (negative control). Elapsed seconds of the closed residual evaluation. |
| Thresholds | Closed residual \(=0\). Open residual \(\ne 0\). No cells/s or wall-time gate. |
| Proves | Closed reflux statement has residual 0. Open statement without reflux is a detectable leftover. Report renderer accepts a PF-08 summary with empty `orders` and a one-node elapsed-time observation. Optional `run_native` wraps AM-09. |
| Does not prove | Native flux-register reflux, MPI gather, multi-patch coverage, average_down, subcycling, interface-band error, timed cells/s. This is not a timed PF run. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
