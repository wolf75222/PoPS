# RB-05 — Sedov off-center (2-d)

Phase 5 robustness contract. Documents the spherical Sedov–Taylor radius
\(R(t)\propto t^{2/5}\) on a 2-d mesh, with the blast centre offset from the
domain. Public 2-d Euler authoring plus a Dim2 `run_native` smoke.

| Field | Content |
|---|---|
| Identifier | `RB-05` |
| `verification_kind` | `robustness` |
| `evidence_status` | `required` |
| Equations | 2-d gamma-law Euler. Strong point explosion in a uniform ambient. Conserved \((\rho,\rho u,\rho v,E)\). No sources after \(t=0\). \(\gamma=1.4\). |
| Oracle | Plan §RB-05 self-similar radius \(R(t)=\xi(E t^2/\rho_0)^{1/(d+2)}\). Spherical geometry \(d=3\) gives \(R(t)\propto t^{2/5}\) on this 2-d mesh (document). Circular \(R(\theta)\) about the off-centre origin. `exact.py` does not read PoPS output. |
| Domain and boundaries | 2-d box \([0,1]^2\). Blast at \((0.4,0.6)\), not the domain midpoint \((0.5,0.5)\). Periodic `uniform_periodic_layout`. |
| Parameters | Dimensionless. \(\rho_0=1\), \(E=1\), \(\xi=1\) for the scaling identity (the numerical \(\xi(\gamma,d)\) is not needed for \(R(t_2)/R(t_1)\)). |
| Native dimensions | `POPS_NATIVE_DIM=2`. `run_native` refuses any other value (no fallback). 3-d spherical remains a later variant. |
| Required capabilities | None on the in-memory path. A later native series needs 2-d Euler + off-centre energy deposit. |
| Configurations | Off-centre circular front vs domain centre. No AMR, no MPI, no \(\Delta x\) series in this increment. |
| Diagnostics | \(R(t)/t^{2/5}\) constant. Task 18 `radial_anisotropy` on \(R(\theta)\). |
| Thresholds | \(R(32)/R(1)=4\). Constant \(R(\theta)\) ⇒ anisotropy \(0\). |
| Proves | Documented 2-d \(R(t)\propto t^{2/5}\). Off-centre origin. Isotropic \(R(\theta)\) has Task 18 anisotropy 0. Report renderer accepts an RB-05 summary. |
| Does not prove | Native shock tracking, the numerical value of \(\xi(\gamma,d)\), energy conservation of a discrete deposit, AMR/MPI symmetry, 3-d spherical campaign, Cartesian fourth-harmonic decay. |
| Resources | Local in-memory contract. No ranks, GPUs, or two-node claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. No compiler, Kokkos, MPI, machine, or Slurm record. |
