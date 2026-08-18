# TM-04 — Larmor / magnetized oscillator

Phase 4 temporal contract. Documents the closed-form cyclotron rotation
\(\mathrm{d}/\mathrm{d}t\,(u_x,u_y)=A(u_x,u_y)\) with
\(A=((0,\omega_c),(-\omega_c,0))\). Exact advance, implicit midpoint, and
explicit Euler stay in memory. A public 1-cell 1-d periodic Case authors the
same rotation as the cell-local source \((\omega_c u_y,-\omega_c u_x)\).
FiniteVolume needs a flux, so the Case carries an inert zero flux; that is
not a private cyclotron integrator. Implicit midpoint has no public factory.
Native compile is optional.

| Field | Content |
|---|---|
| Identifier | `TM-04` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | \(\dot u_x=\omega_c u_y\), \(\dot u_y=-\omega_c u_x\). Skew generator; \(\lvert u\rvert\) is an invariant of the exact flow and of implicit midpoint. |
| Oracle | Rotation matrix by \(\theta=\omega_c t\): \(R(\theta)=((\cos\theta,\sin\theta),(-\sin\theta,\cos\theta))\). Canonical \(\omega_c=1\), \(u_0=(1,0)\). `exact.py` does not read PoPS output. |
| Domain and boundaries | In-memory path: 0-d velocity plane \((u_x,u_y)\). Public Case: 1-d periodic unit interval, 1 cell. |
| Parameters | Dimensionless. \(\omega_c=1\). One-step size \(\Delta t=0.25\). |
| Native dimensions | `POPS_NATIVE_DIM=1` only. Selected by the public resolve/compile path. |
| Required capabilities | None on the in-memory path. Public Case: Cartesian uniform periodic, public cyclotron source, inert zero flux + first-order + Rusanov, SSPRK2 + `FixedDt`. KokkosSerial. MPI off. |
| Configurations | Exact rotation vs implicit midpoint vs explicit Euler. Public Case on uniform \(N=1\). No AMR, no \(\Delta t\) series in this increment. |
| Diagnostics | Euclidean speed \(\lvert u\rvert\). Period return at \(t=2\pi/\omega_c\). One-step implicit midpoint vs explicit Euler. |
| Thresholds | Exact \(\lvert u\rvert\) conserved (machine). Exact return to \(u_0\) at one period. Implicit midpoint \(\lvert u\rvert\) conserved to \(10^{-12}\) on one step. Explicit Euler \(\lvert u\rvert\) grows. |
| Proves | Closed-form Larmor rotation conserves speed and is \(2\pi/\omega_c\)-periodic. Implicit midpoint is unitary on the cyclotron generator. Explicit Euler is not. Public cyclotron-source Case resolves in Dim1 without compile. Report renderer accepts a TM-04 summary. |
| Does not prove | Native Lorentz-source instrumentation, compiled implicit-midpoint unitarity, Boris / implicit-Lorentz order, energy conservation of a compiled Program, AMR, MPI, performance. |
| Resources | Local in-memory contract. Public Case is 1-d, 1 cell. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compile records stay on the optional `run_native` path. |
