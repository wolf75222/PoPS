# TM-02 — noncommuting Strang

Phase 4 temporal-splitting case. Manufactured 2×2 linear pair
\(A=\mathrm{diag}(a_1,a_2)\), \(B=\begin{pmatrix}-\nu&\nu\\\nu&-\nu\end{pmatrix}\)
with \(a_1\neq a_2\) so \(AB\neq BA\). Exact in-memory 0-d subflows; Lie is order 1
and Strang is order 2. The public Case authors the plan PDE
\(\partial_t q + A\partial_x q = B q\) on a 1-d periodic mesh: flux
\((a_1 q_0, a_2 q_1)\), waves \((a_1,a_2)\), collision \(B\) as a local linear
operator, and official `pops.lib.time.Strang` / `Lie` as
\(C/2\)–\(T\)–\(C/2\). Collision is the exact \(\exp(B\alpha)\) map (one
Euler step on \(K=(\exp(B\alpha)-I)/\alpha\)). Transport is Heun RK2. `pops.lib.time.Strang`
only declares StagePoint partitions `first`/`second`; FiniteVolume looks up
`explicit`, so the custom Programs alias `explicit` onto the transport clock.
State components are `q0`,`q1` (not `u`,`v`, which infer `velocity:1` and fail
Dim1 compile). The order campaign uses WENO5 so spatial error does not mask
the temporal orders. Native compile is optional and does not call ROMEO.

| Field | Content |
|---|---|
| Identifier | `TM-02` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | In-memory: \(\dot u=(A+B)u\). Public Case: \(\partial_t q + A\partial_x q = B q\). \(A=\mathrm{diag}(a_1,a_2)\). \(B=\begin{pmatrix}-\nu&\nu\\\nu&-\nu\end{pmatrix}\). Lie \(\Phi^B_{\Delta t}\circ\Phi^A_{\Delta t}\). Strang \(\Phi^C_{\Delta t/2}\circ\Phi^T_{\Delta t}\circ\Phi^C_{\Delta t/2}\). |
| Oracle | Exact combined flow \(\exp((A+B)t)u_0\). Exact subflows \(\exp(At)\), \(\exp(Bt)\). Optional spatial Fourier mode \(\exp((B-ikA)t)\hat q\). Loaded with `load_sibling_module`. `exact.py` does not read PoPS output. |
| Domain and boundaries | In-memory path: 0-d / two-component ODE. Public Case: 1-d periodic unit interval, default \(n\ge 16\) cells. `resolve_plan` also accepts \(n=1\). |
| Parameters | Dimensionless. Canonical \(a_1=1\), \(a_2=3\), \(\nu=1\), \(u_0=(1,0)\), \(T=1\). Temporal series \(\Delta t,\Delta t/2,\Delta t/4,\Delta t/8\) with \(\Delta t=0.1\). |
| Native dimensions | `POPS_NATIVE_DIM=1` only. Selected by the public resolve/compile path. |
| Required capabilities | None on the in-memory path. Public Case: Cartesian uniform periodic, official `pops.lib.time.Strang`/`Lie` + `FixedDt`. KokkosSerial. MPI off. |
| Configurations | Manufactured noncommuting pair. Formal Lie order 1, Strang order 2. Spatial C/2–T–C/2 split. No AMR. |
| Diagnostics | Task 15 observed order on each \(\Delta t\) series (`kind=temporal`, variables `lie` and `strang`). Commutator \(AB-BA\). |
| Thresholds | Lie observed order \(\ge 0.8\). Strang observed order \(\ge 1.8\) (plan §8.2 Strang). In-memory \(AB\neq BA\) when \(a_1\neq a_2\). |
| Proves | \(A\) and \(B\) do not commute. Manufactured Lie temporal order 1. Manufactured Strang temporal order 2. Public spatial Strang Case resolves in Dim1 for \(n=1\) and \(n\ge 16\) without compile. Report renderer accepts a TM-02 temporal summary. |
| Does not prove | Native Lie/Strang Program order until a compiler series is run. TM-03 collision relaxation, TM-08 reversibility, AMR, Poisson, coupling, MPI, performance. |
| Resources | Local in-memory \(\Delta t\) series. Public Case is 1-d, default 16 cells. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compile records stay on the optional `run_native` path. |
