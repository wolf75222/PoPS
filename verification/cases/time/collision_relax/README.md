# TM-03 — exact collision relaxation

Phase 4 temporal contract. Documents the closed-form BGK-style relaxation
\(\mathrm{d}u/\mathrm{d}t=-\nu(u-\bar u)\) and the two-species drag
\(\rho_1\mathrm{d}q_0/\mathrm{d}t=K(q_1-q_0)\),
\(\rho_2\mathrm{d}q_1/\mathrm{d}t=K(q_0-q_1)\). The in-memory maps
`run.relax` and `run.relax_two_species` are exact. A public scalar Case
authors the cell-local source \(-\nu(u-\bar u_{\mathrm{IC}})\) toward the
manufactured IC barycenter (a constant). A public two-component Case
authors the coupled source \(((K/\rho_1)(q_1-q_0),(K/\rho_2)(q_0-q_1))\)
with an inert spatial flux. FiniteVolume needs a flux; that is not a
private collision operator. Native compile is optional.

| Field | Content |
|---|---|
| Identifier | `TM-03` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | Scalar: \(\mathrm{d}u/\mathrm{d}t=-\nu(u-\bar u)\). \(\bar u\) is the volume-weighted barycenter of \(u\). Two-species: \(\rho_1\mathrm{d}q_0/\mathrm{d}t=K(q_1-q_0)\), \(\rho_2\mathrm{d}q_1/\mathrm{d}t=K(q_0-q_1)\). \(V=(\rho_1 q_0+\rho_2 q_1)/(\rho_1+\rho_2)\) is constant; slip \(w=q_0-q_1\) decays as \(w(0)e^{-\lambda t}\) with \(\lambda=K(1/\rho_1+1/\rho_2)\). No spatial flux. |
| Oracle | Closed form \(u(t)=\bar u+(u_0-\bar u)e^{-\nu t}\). Manufactured \(u_0=1+\frac12\cos(2\pi x)\) on the unit interval so \(\bar u=1\). Two-species map `run.relax_two_species`. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\) for the manufactured field. The ODE itself is cell-local. |
| Parameters | Dimensionless. \(\nu=2\). Two-species defaults \(K=\rho_1=\rho_2=1\) so \(\lambda=2\). Uniform \(N=32\) cells on the scalar in-memory path. |
| Native dimensions | `POPS_NATIVE_DIM=1` only. Selected by the public resolve/compile path. |
| Required capabilities | Cartesian uniform periodic. KokkosSerial. MPI off. Public scalar source \(-\nu(u-\bar u_{\mathrm{IC}})\), inert zero flux + MUSCL/VanLeer + ScalarUpwind, SSPRK2 + `FixedDt`. Public two-species source + inert flux + FirstOrder/Rusanov + SSPRK2. |
| Configurations | Exact map vs barycenter invariant. Public scalar Case on uniform \(N=32\). Public two-species Case (`q0`,`q1`). Formal temporal order 2. No AMR. |
| Diagnostics | Pointwise exponential identity. Volume-weighted barycenter of \(u(t)\) equals \(\bar u\). Two-species \(V\) constant and slip exponential. |
| Thresholds | \(u(0)=u_0\). Deviation halves at \(t=\ln 2/\nu\). Barycenter drift is 0 (machine). Two-species slip halves at \(t=\ln 2/\lambda\). |
| Proves | Closed-form exponential relaxation. Conserved barycenter moment. `run.relax` matches `exact_relax`. Public scalar Case resolves in Dim1 without compile. Public two-species Case (`two_species_resolve`) resolves in Dim1 without compile. Report renderer accepts a TM-03 summary. |
| Does not prove | Native temporal order until a compiler series is run. Live nonlocal barycenter collision. BGK Maxwellian matching. IMEX/Strang coupling. AMR, MPI, performance. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compile records stay on the optional `run_native` path. |
