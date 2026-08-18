# TR-02 — Transported Gaussian pulse

Phase 1 localized scalar advection. Exact solution is a periodic translation.
This worktree does not catalogue the case in `verification/manifest.toml`.

| Field | Content |
|---|---|
| Identifier | `TR-02` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | \(\partial_t q + a\partial_x q = 0\) with \(q(x,0)=q_0+A\exp(-\|x-x_0\|^2/(2\sigma^2))\). Canonical \(a=1\), \(q_0=0\), \(A=1\), \(x_0=0.37\), \(\sigma=0.08\). |
| Oracle | `translated_gaussian`: minimum-image translation on periodic \([0,1]\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). |
| Parameters | Default in-memory grid \(n=32\). Resolutions for a later order series: \(N=16,32,64,128\). |
| Native dimensions | `POPS_NATIVE_DIM=1` only. In-memory report does not load a native artifact. |
| Required capabilities | Public 1-d scalar advection authoring (`CartesianDomain` + `Cartesian1D`, MUSCL/VanLeer + ScalarUpwind, SSPRK2). KokkosSerial listed; MPI off. |
| Configurations | Uniform periodic. Optional `run_native` uses `BindArray` IC sampled from `exact_gaussian`. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞; pulse mass; periodic barycenter of \(q-q_0\); max location (native later). |
| Thresholds | In-memory exact vs exact: \(L^\infty=0\). Native spatial order is out of scope here. |
| Proves | Translation identity, periodic barycenter motion, mass independence of the exact field, public `resolve_plan` without compile, schema-valid campaign report. |
| Does not prove | Observed spatial order, AMR, Poisson, coupling, MPI, performance. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ROMEO job. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. |
