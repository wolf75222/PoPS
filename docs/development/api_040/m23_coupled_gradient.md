# M23 reversible component-gradient law, periodic Cartesian realization

The normative mathematical source is the preserved handoff file
`PoPS_Codex_handoff_0.4.0/reference/PoPS_API_v0.4.0/sources/mathematical_original.tex`
in the primary PoPS checkout, section *Gradient-dependent extensions beyond
diffusion*, lines 3067–3083. Its closed classification witness is

\[
  w_t=\eta_H J w_{xx},\qquad
  J=\begin{pmatrix}0&-1\\1&0\end{pmatrix},\quad \eta_H=0.3.
\]

For a Fourier mode with wave number 2, the continuous phase at time 0.2 is
\(-\eta_H 2^2 0.2=-0.24\); its continuous quadratic norm is constant. The
older revision test used the opposite rotation sign and inspected only the
norm, so it cannot establish this orientation.

`Model.coupled_gradient_flux` declares separate, exact component matrices
\(D=D^T\succeq0\) and \(R=-R^T\). The signed *rate* flux is
\((D+R)\nabla U\), equivalent to dissipative flux \(D\nabla U\) minus physical
reversible flux \(-R\nabla U\). The declaration supports any state component
count and Cartesian frame and stores both matrices in the physical identity;
this initial constitutive subtype requires finite constant entries. It is not
relabelled as `TensorDiffusion` or checked by that method's SPD Jacobian rule.
The PSD authoring check uses exact rational arithmetic on the represented
integer/binary-float entries: rank-deficient positive matrices are admitted,
and a genuinely negative mode is refused without clipping or tolerance.

`CoupledGradient` selects a distinct `PreparedCoupledGradient` provider. Its
constant-matrix realization covers one, two, or three periodic Cartesian axes
on Uniform storage, with first-order cell means and a two-point
component-coupled gradient on each axis. Nonperiodic traces, variable matrices,
AMR composite transfer, implicit stages, and transport composition remain implementation obligations, not
restrictions on the underlying physical equation. These routes refuse before
native publication. Every accepted face occurrence is recorded in the same
conservative stage ledger as other spatial fluxes.
The method's serialized `schema_version: 1` and
`periodic_two_point_component_matrix_v1` identifier remain valid: neither the
constitutive matrix nor the face formula changed; the supported native
dimension now follows the existing Cartesian axis sum.
Positive, repeated occurrences of the *same exact* flux are admitted: the
constitutive face is prepared once, the RHS uses the sum of coefficients, and
the accepted ledger retains one occurrence identity and temporal weight per
use. Nonpositive or foreign-law uses are refused.

On a periodic Cartesian grid, the native semidiscrete Fourier symbol is
\(-4\sum_d\sin^2(k_dh_d/2)/h_d^2\). The physical identity spatial tensor makes
the tensor-face construction exactly \(F_{d,i+1/2}=(D+R)(U_{i+1}-U_i)/h_d\),
even when spacings differ by axis. With cell volume \(V\), its quadratic energy
obeys \(\langle U,L(U)\rangle_V=-\sum_{d,faces}V\,\Delta_dU^TD\Delta_dU/h_d^2\leq0\):
the skew part contributes exactly zero on the periodic mesh. The native
reported frequency is a finite operator-magnitude bound
\(4\lVert D+R\rVert_\infty\sum_d h_d^{-2}\), not a monotonicity or temporal
stability certificate. The selected SSPRK2 witness has per-step
complex amplification \(1-z^2/2-iz\), where
\(z=\eta_H[4\sin^2(kh/2)/h^2]\Delta t\). Its modulus exceeds one for every
nonzero imaginary-axis mode; this witness verifies the measured amplification
and phase, not general SSPRK2 stability for Hall evolution. No monotone
diffusion CFL is borrowed from `TensorDiffusion`.

`examples/migration/scientific/api040_m23_hall_fourier.py` fixes the grid and
acceptance thresholds before native execution: N=32,64, exact cell averages of
\((1,0.2)\cos(2x)\), both component orders, Hall-zero control, \(\Delta t=0.002\),
100 SSPRK2 steps to 0.2. It saves native initial/final states as NPZ and
reopens them before comparing initial means, final complex amplitude, signed
phase, norm, and time to the independently written discrete oracle. The
predeclared error limits are 3e-14 initially, 3e-10 for state/phase/norm, and
3e-14 for time. One accepted step inventory and one receipt are required for
each actual run; all ranks receive the same verdict. MPI compilation uses the
authenticated single-publisher helper. Each installed receipt qualifies only
its exact source, SDK and native dimension.

Source-only checks: `test_coupled_gradient_hall_source.py` covers physical
identity, off-diagonal sign, provider selection, accepted-face publication,
component permutation and Hall zero. The independent review uses a separate
constant-matrix stencil/energy oracle, including three components. Neither
source generation nor that oracle is a native numerical result. The separate
Dim2 Fourier/ledger witness uses unequal axis spacings, both component
permutations and every axis/side incidence; its installed native execution is
required to qualify this extended route. Dim3 currently has source and
complete-C++ syntax coverage, not an installed numerical trajectory.
