# CP-10 — Jeans stable / unstable

Linear self-gravity of an isothermal fluid about a uniform rest state.
Dispersion \(\omega^2=c_s^2 k^2-4\pi G\rho_0\). Attractive gravity is the
minus sign: an overdensity is a potential well (\(\partial_{xx}\phi=4\pi G\,\delta\rho\),
\(g=-\partial_x\phi\)). Native compile is optional.

| Field | Content |
|---|---|
| Identifier | `CP-10` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | Linearized continuity \(\partial_t\delta\rho+\rho_0\partial_x u=0\). Momentum \(\partial_t u+(c_s^2/\rho_0)\partial_x\delta\rho=g\) with \(g=-\partial_x\phi\). Attractive Poisson \(\partial_{xx}\phi=4\pi G\,\delta\rho\). |
| Oracle | \(\omega^2=c_s^2 k^2-4\pi G\rho_0\). \(k_J=\sqrt{4\pi G\rho_0}/c_s\). Closed field \(U=\bar U+\epsilon\Re(r\exp(ikx-i\omega t))\) when \(k>k_J\); growing \(U=\bar U+\epsilon\Re(r\exp(ikx+\gamma t))\) with \(\gamma=\lvert\omega\rvert\) when \(k<k_J\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic interval \([0,4\pi]\) so both \(k=2\) and \(k=0.5\) close. |
| Parameters | \(c_s=1\), \(4\pi G\rho_0=1\Rightarrow k_J=1\). \(\bar U=(\rho_0,0)=(1,0)\), \(\epsilon=10^{-4}\). Stable \(k=2\) (\(\omega=\sqrt{3}\)). Unstable \(k=0.5\) (\(\gamma=\sqrt{3}/2\)). Dimensionless. |
| Native dimensions | `POPS_NATIVE_DIM=1`. In-memory path does not load a native artifact. |
| Required capabilities | Cartesian 1-d, uniform, periodic, KokkosSerial, MPI off. |
| Configurations | Single resolution \(n=32\) for the in-memory report. Authored scheme: Rusanov, MUSCL/VanLeer, SSPRK2, AdaptiveCFL. Self-gravity stays on the oracle. |
| Diagnostics | Sign of \(\omega^2\) at \(k=2\) and \(k=0.5\). Growth factor \(\exp(\gamma t)\) at \(t=1\) for \(k=0.5\). Task 2 volume-weighted L1/L2/L∞ on density of exact vs exact. |
| Thresholds | Attractive residual \(\lvert\omega^2-(c_s^2 k^2-4\pi G\rho_0)\rvert\le 10^{-12}\). \(\omega^2(k=2)>0\), \(\omega^2(k=0.5)<0\). Growth mismatch \(\le 10^{-12}\). Exact-vs-exact L∞ = 0. No spatial-order gate on the in-memory path. |
| Proves | Attractive Jeans dispersion identity; \(k>k_J\) oscillates and \(k<k_J\) grows as \(\exp(\gamma t)\); report renderer. |
| Does not prove | Native \(\omega_{\mathrm{num}}(k)\) or \(\gamma_{\mathrm{num}}\), Jeans collapse into the nonlinear regime, AMR, MPI, 2-d/3-d filaments, cosmological expansion. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
