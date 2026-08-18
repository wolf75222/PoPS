# TR-06 — axis permutation / reflection

Phase 3 leftover transport contract. Manufactured 2-d product of TR-01 sines
on the periodic unit square. After an \(x\leftrightarrow y\) swap or a periodic
reflection \(x\mapsto 1-x\), remapped exact fields are identical. In-memory
only; no live runtime, compile, or ROMEO.

| Field | Content |
|---|---|
| Identifier | `TR-06` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | \(\partial_t q + a_x\partial_x q + a_y\partial_y q = 0\). Conservative scalar \(q\). Canonical \((a_x,a_y)=(1,1/2)\). |
| Oracle | Product of TR-01 sines \(q(x,y,t)=q_0+\varepsilon\sin(2\pi k_x(x-a_x t))\sin(2\pi k_y(y-a_y t))\) with \(q_0=1\), \(\varepsilon=10^{-2}\), \((k_x,k_y)=(1,2)\). Loaded with `load_sibling_module` from `verification/cases/transport/advection_sine/exact.py`. `exact.py` does not read PoPS output. |
| Domain and boundaries | 2-d periodic unit square \([0,1]^2\). Samples are uniform cell centres, axis 0 is \(x\). |
| Parameters | Dimensionless. Default grid \(N=32\). Evaluation time \(T=0.125\) (not a reflection-symmetric phase of the \(x\)-sine). Anisotropic \((k,a)\) so the unmapped swapped / reflected fields differ. |
| Native dimensions | `POPS_NATIVE_DIM=2` listed for the planner. This increment does not load a native artifact. |
| Required capabilities | None on the in-memory path. A later native series needs a 2-d periodic Cartesian layout and axis-permuted / reflected copies. |
| Configurations | Uniform square cells. Exact coordinate maps only. No AMR, no \(\Delta t\) series in this increment. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ of the remapped field vs the original product. Unmapped swapped / reflected arrays must differ. |
| Thresholds | After mapping, L1 = L2 = L∞ = 0. |
| Proves | TR-01 oracle reuse via `load_sibling_module`. Exact \(x\leftrightarrow y\) and \(x\mapsto 1-x\) identities. Report renderer accepts a TR-06 summary. |
| Does not prove | Native 2-d advection, discrete axis permutation of a compiled Program, ghost fill, AMR, Poisson, coupling, MPI, performance. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
