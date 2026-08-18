# CP-02 — Cold Langmuir wave (1-d)

v1.5 code-verification of a cold-electron standing Langmuir eigenmode on a
periodic unit interval. The public pipeline is Case → validate → resolve →
compile → bind → `pops.run`. Cell-average ICs and oracles follow plan §7.3.
Fluid `|u|` is not the time-step constraint; `FixedDt` uses the phase speed
`ω_pe / k`.

| Field | Content |
|---|---|
| Identifier | `CP-02` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | Cold electrons: `∂_t n_e + ∂_x(n_e u_e) = 0`, `∂_t(n_e u_e) + ∂_x(n_e u_e²) = (q_e / m_e) n_e E` with `q_e = -e`. Fixed ions `n_i = n_0`. Gauss `∂_x E = e (n_i - n_e) / ε_0`. `E = -∂_x φ`. |
| Oracle | Closed 1-d standing wave plus the same fields as cell averages. `exact.py` does not import `pops` or read PoPS output. `linear_eigenmode_fields` rebuilds that standing wave from the linearized cold-fluid eigenvector of *this* PDE (`Re[r(x) e^{-iωt}]`) and cross-checks the closed form. PoPS does not auto-generate `r` from a generic matrix. |
| Domain and boundaries | 1-d periodic unit interval `[0, 1]`. |
| Parameters | `e = m_e = ε_0 = n_0 = n_i = 1 ⇒ ω_pe = 1`. `k = 2π`. `A = 10^{-4}`. Canonical `t_end` is one period `2π`. |
| Native dimensions | `POPS_NATIVE_DIM = 1`. No fallback to another rank. |
| Required capabilities | Cartesian 1-d, uniform, periodic, FFT Poisson, electrostatic source at every SSPRK2 stage, KokkosSerial, MPI off. OpenMP and MPI are bounded smokes: missing capability is a required failure, not a silent fallback. |
| Configurations | Acceptance reconstruction is public `WENO5-Z`. Serial **global** series `n = 16, 32, 64, 128` at constant phase CFL `0.4`. A separated **temporal** series holds `N` fixed and varies `dt`. Isolated **spatial** (`dt ∝ h²`) is optional and separately labeled. MUSCL+VanLeer is a non-acceptance TVD variant. |
| Diagnostics | `E_ω` and `H_2` from a real probe time series (FFT, phase fit, zero crossings). L1/L2/L∞ on cell-average `n`, `φ`, `E`. Mass, charge, kinetic and electrostatic energy. |
| Thresholds | §9.3 last two L∞ intervals above `1e-14` must be `≥ 1.8`. Finite-only is not a pass. Exact-vs-exact is refused. |
| Proves | Cold Langmuir eigenmode with the documented Poisson sign; frequency/phase/amplitude from native time series; global order of the coupled SSPRK2 + FFT scheme; energy exchange KE ↔ electrostatic. |
| Does not prove | Isolated spatial order, warm dispersion (CP-03), 2-d oblique modes, AMR, GPU, or WarpX kinetic Langmuir. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. Account `r250127`, x64cpu first, ≤ 2 nodes. |
| Provenance | `repository_sha = git rev-parse HEAD`. Live MPI ranks and Kokkos concurrency overwrite manifest defaults. Leaf digest is the installed exact-rank `_pops.so`. |

## Sign contract

Do not flip Poisson in analyze. With `q_e = -e` and `∂_x E = e (n_i - n_e) / ε_0`:

```
n_e = n_0 + A cos(kx) cos(ω_pe t)
u_e = (A ω_pe)/(n_0 k) sin(kx) sin(ω_pe t)
E   = -(e A)/(ε_0 k) sin(kx) cos(ω_pe t)
φ   = -(e A)/(ε_0 k²) cos(kx) cos(ω_pe t)
```

A sign mismatch between `linear_eigenmode_fields` and this closed form is a
coupling failure. PoPS does not auto-generate the eigenvector.

## Typical errors

| Symptom | Likely cause |
|---|---|
| Frequency right, amplitude wrong | Dissipation, energy source, time centering |
| Energy drift | Field not updated at every RK stage, `J·E` |
| Order 1 on Langmuir, order 2 on advection | Field/source frozen across stages |
| Exact-vs-exact L∞ = 0 | Oracle compared to itself; not evidence |
