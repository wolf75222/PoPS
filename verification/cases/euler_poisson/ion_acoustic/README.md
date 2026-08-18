# CP-06 — ion-acoustic eigenmode

Two-fluid 1-d toy: Boltzmann electrons, cold ions. Fourier symbol \(M(k)\)
on \((\delta n_i,\delta u_i)\) has known eigenpairs. The closed reference is
\(U=\bar U+\epsilon\Re(r\exp(ikx-i\omega t))\). Matching native physics needs
screened Poisson; uniform FFT/CartesianCG do not implement it. The authored
Case is the local unscreened ion-sound limit. ``run_native`` compiles and
advances that authored Case. Screened Debye stays on the ``exact.py`` oracle.

| Field | Content |
|---|---|
| Identifier | `CP-06` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | Cold ions \(\partial_t n_i+\partial_x(n_i u_i)=0\), \(\partial_t u_i+u_i\partial_x u_i=(e/m_i)E\). Boltzmann electrons \(n_e=n_0\exp(e\phi/T_e)\). Poisson \(-\varepsilon_0\partial_{xx}\phi=e(n_i-n_e)\), \(E=-\partial_x\phi\). Linearized Fourier symbol \(M(k)=\begin{pmatrix}0&-ikn_0\\-ik(c_s^2/n_0)/(1+k^2\lambda_D^2)&0\end{pmatrix}\). Authored public flux is the unscreened limit \(\partial_t n+n_0\partial_x u=0\), \(\partial_t u+(c_s^2/n_0)\partial_x n=0\). |
| Oracle | Dispersion \(\omega^2=k^2 c_s^2/(1+k^2\lambda_D^2)\) with \(c_s^2=T_e/m_i\), \(\lambda_D^2=\varepsilon_0 T_e/(n_0 e^2)\). Eigenpairs \(\lambda_\pm=\mp i\omega\), \(r_+=(1,\omega/(kn_0))\), \(r_-=(1,-\omega/(kn_0))\). Closed field \(U(x,t)=\bar U+\epsilon\Re(r\exp(ikx+\lambda t))\). Plus branch is \(\exp(ikx-i\omega t)\). `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). |
| Parameters | \(e=m_i=\varepsilon_0=n_0=T_e=1\), \(\bar U=(n_0,0)\), \(\epsilon=10^{-4}\), \(k=2\pi\). Sweep \(k/2\pi=1,2,4,8\). Dimensionless. |
| Native dimensions | `POPS_NATIVE_DIM=1`. `run_native` returns the unscreened ion-sound state as shape `(2, n)`. |
| Required capabilities | Cartesian 1-d, uniform, periodic, KokkosSerial, MPI off. |
| Configurations | Single resolution \(n=32\) for the in-memory report. Authored scheme: Rusanov, MUSCL/VanLeer, SSPRK2, AdaptiveCFL. BindArray IC from `exact_state(..., t=0, mode=)`. Native compile/bind/run advances the unscreened ion-sound Case. Screened Poisson stays on the `exact.py` oracle. No second dynamic species. |
| Diagnostics | Dispersion residual \(\omega^2-k^2 c_s^2/(1+k^2\lambda_D^2)\). Eigenvector residual \(\|Mr+i\omega r\|_\infty\) on the plus branch and \(\|Mr-\lambda r\|_\infty\) on both modes. Closed-form time advance, including \(\exp(Mt)\hat U\). Task 2 volume-weighted L1/L2/L∞ on density of exact vs exact. |
| Thresholds | Dispersion residual \(\le 10^{-12}\). Eigenvector residual \(\le 10^{-12}\). Time-advance mismatch \(\le 10^{-12}\). Exact-vs-exact L∞ = 0. No spatial-order gate on the in-memory path. |
| Proves | Ion-acoustic \(\omega(k)\); \(Mr=-i\omega r\); closed time advance \(e^{\lambda t}\); report renderer; public 1-d periodic Case validates and resolves; optional native compile of the unscreened ion-sound limit. |
| Does not prove | Native screened two-fluid Euler–Poisson, measured \(\omega_{\mathrm{num}}(k)\), warm-ion / collisional / magnetized branches, modal contamination order, AMR, MPI, DSL/native parity. Uniform FFT/CartesianCG do not implement screened Poisson. Native compile requires Kokkos + a compiler. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
