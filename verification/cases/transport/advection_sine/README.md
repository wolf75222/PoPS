# TR-01 — 1-d periodic advection sine

Phase 1 first scientific case. Manufactured translation of a sine on the
periodic unit interval. Native compile is optional; the in-memory path does not
call a solver.

| Field | Content |
|---|---|
| Identifier | `TR-01` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | \(\partial_t q + a \partial_x q = 0\). Conservative scalar \(q\). No sources. Canonical \(a=1\). |
| Oracle | Manufactured translation \(q(x,t)=q(x-at,0)\) of \(q=q_0+\varepsilon\sin(2\pi k x)\) with \(q_0=1\), \(\varepsilon=10^{-2}\), \(k=1\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). `POPS_NATIVE_DIM=1`. |
| Parameters | Dimensionless. \(T=1\) one period. Resolutions \(N=16,32,64,128\) (256 later / ROMEO). |
| Native dimensions | `POPS_NATIVE_DIM=1` only. Selected by the public resolve/compile path. |
| Required capabilities | Cartesian uniform periodic. KokkosSerial. MPI off. MUSCL/VanLeer + ScalarUpwind, SSPRK2. |
| Configurations | Uniform 1-d cells. Adaptive CFL \(0.45\). Formal spatial order 2. No AMR. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ vs the manufactured sine. Task 15 observed order on a resolution series. |
| Thresholds | Observed order \(\ge 1.8\) when a native series exists. In-memory exact-vs-exact L∞ is 0. |
| Proves | Periodic translation identity of the oracle. Public Case authoring resolves in Dim1 without compile. Report renderer accepts a TR-01 summary. |
| Does not prove | Native spatial order until a compiler series is run. AMR, Poisson, coupling, MPI, performance. |
| Resources | Local 1-d series. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compile records stay on the optional `run_native` path. |
