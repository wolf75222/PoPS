# TM-07 — field update every RK stage

Phase 4-adjacent coupling-time contract. A field-coupled explicit RK
program must solve or update the field at every required stage. The
in-memory helpers document SSPRK2 = 2 and SSPRK3 = 3 field solves per
step. The public Case is CP-02 cold Langmuir with `SSPRK2(..., fields=)`
so Poisson is attached at each stage. Native compile is optional.

| Field | Content |
|---|---|
| Identifier | `TM-07` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | Coupled hyperbolic + elliptic field. The temporal Program is explicit SSPRK2 (2 stages) or SSPRK3 (3 stages). The field used by a stage source must be the field of that stage state. Public Case: cold electrons, fixed ions, FFT Poisson, `q_e n E / m_e` source. |
| Oracle | Documented stage counts: SSPRK2 = 2, SSPRK3 = 3. `required_field_solves(stages) = stages`. Frozen-field (1 solve/step, held across stages) is the negative control. `exact.py` does not read PoPS output. |
| Domain and boundaries | Contract is integrator-local on the in-memory path. Public Case: 1-d periodic unit interval `[0,1]`. |
| Parameters | Dimensionless. Integrators `SSPRK2`, `SSPRK3`. Frozen-field flag is the anti-pattern. Public Case uses CP-02 units \(e=m_e=\varepsilon_0=n_0=1\). |
| Native dimensions | `POPS_NATIVE_DIM=1`. Selected by the public resolve/compile path. |
| Required capabilities | None on the in-memory path. Public Case: Cartesian 1-d, uniform, periodic, Poisson, electrostatic source, KokkosSerial, MPI off. |
| Configurations | Per-stage field update vs frozen-field. Public Case authors `SSPRK2(..., fields=)`. No AMR, no \(\Delta t\) series in this increment. |
| Diagnostics | Field-solve count per step. Frozen-field count is strictly less than the per-stage count. Optional native run returns conserved `(n, n u)` of shape `(2, n)`. |
| Thresholds | SSPRK2 → 2 solves/step. SSPRK3 → 3 solves/step. Frozen-field → 1 solve/step. |
| Proves | Documented SSPRK2/SSPRK3 stage counts. `required_field_solves(stages)=stages`. Frozen-field is a detectable under-solve. Public `SSPRK2(..., fields=)` Case resolves. Report renderer accepts a TM-07 summary. |
| Does not prove | Runtime field-solve instrumentation (the public API does not expose a stage counter), coupled temporal order, CP-01 / CP-02 frequency order, AMR, MPI, performance. Call-count alone is not the full plan proof (plan §TM-07). |
| Resources | Local in-memory contract plus optional 1-d native run. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compile records stay on the optional `run_native` path. |
