# TM-05 — asymptotic-preserving limit

Phase 4 temporal AP-limit contract. Documents the toy IMEX relaxation
\(\mathrm{d}y/\mathrm{d}t=-(y-g)/\varepsilon+f\) with \(g=0\), \(f=0\),
\(y(0)=1\). Backward Euler stays stable as \(\varepsilon\to 0\); forward
Euler blows when \(\varepsilon\ll\Delta t\). In-memory helpers stay. A public
1-cell 1-d periodic Case authors \(L=-1/\varepsilon\) and
`pops.lib.time.IMEX` (Euler tableau, DenseLU). Native compile is optional
and does not call ROMEO.

| Field | Content |
|---|---|
| Identifier | `TM-05` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | \(\mathrm{d}y/\mathrm{d}t=-(y-g)/\varepsilon+f\). Canonical \(g=0\), \(f=0\). No spatial flux. Implicit backward Euler \(y^{n+1}=(y^n+(\Delta t/\varepsilon)g+\Delta t\,f)/(1+\Delta t/\varepsilon)\). Explicit Euler \(y^{n+1}=y^n+\Delta t\,(-(y^n-g)/\varepsilon+f)\). Public Case: inert zero flux + local linear \(L=-1/\varepsilon\). IMEX Euler + DenseLU is that backward-Euler map. |
| Oracle | Closed form \(y(t)=e^{-t/\varepsilon}\). Reduced limit \(\varepsilon\to 0\) is \(y=0\). `exact.py` does not read PoPS output. |
| Domain and boundaries | In-memory path: 0-d scalar ODE. Public Case: 1-d periodic unit interval, 1 cell. |
| Parameters | Dimensionless. \(y(0)=1\). Macro step \(\Delta t=0.1\) fixed. Stiffness sweep \(\varepsilon=1,10^{-1},10^{-2},10^{-3},10^{-4}\). Public Case defaults to \(\varepsilon=10^{-4}\). |
| Native dimensions | `POPS_NATIVE_DIM=1` only. Selected by the public resolve/compile path. |
| Required capabilities | None on the in-memory path. Public Case: Cartesian uniform periodic, `pops.lib.time.IMEX` + `FixedDt`. KokkosSerial. MPI off. |
| Configurations | One macro step at fixed \(\Delta t\). Implicit AP vs explicit Euler. No AMR, no \(\Delta t\) series in this increment. |
| Diagnostics | \(\lvert y\rvert\) after one macro step. Implicit monotone approach to the reduced limit. Explicit amplification when \(\varepsilon\ll\Delta t\). |
| Thresholds | Implicit \(\lvert y\rvert\le 1\) for every \(\varepsilon\). Explicit \(\lvert y\rvert>10^{2}\) (or non-finite) at \(\varepsilon=10^{-4}\). Implicit \(\lvert y\rvert\to 0\) as \(\varepsilon\to 0\) (atol \(2\cdot 10^{-3}\) at \(\varepsilon=10^{-4}\)). |
| Proves | Closed-form exponential relaxation. Implicit backward Euler is uniformly stable and captures \(y=0\) as \(\varepsilon\to 0\). Explicit Euler diverges for \(\varepsilon=10^{-4}\). Public IMEX Euler Case resolves in Dim1 without compile. Report renderer accepts a TM-05 summary. |
| Does not prove | Native IMEX Program AP property until a compiler step is run. Spatial transport+source IMEX, TM-03 collision map, AMR, MPI, performance. |
| Resources | Local in-memory stiffness sweep. Public Case is 1-d, 1 cell. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compile records stay on the optional `run_native` path. |
