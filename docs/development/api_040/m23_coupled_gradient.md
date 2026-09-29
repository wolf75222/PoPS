# M23 reversible component-gradient law, first native realization

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

`CoupledGradient` selects a distinct `PreparedCoupledGradient` provider. Version
1 realizes only one periodic Cartesian axis on Uniform storage, with first
order cell means and the two-point component-coupled gradient. Other spatial
dimensions, traces, variable matrices, AMR composite transfer, implicit
stages, and transport composition remain implementation obligations, not
restrictions on the underlying physical equation. These routes refuse before
native publication. Every accepted face occurrence is recorded in the same
conservative stage ledger as other spatial fluxes.

For a periodic grid of spacing \(h=2\pi/N\), the native semidiscrete Fourier
symbol is \(-4\sin^2(kh/2)/h^2\). The skew part contributes zero to the
semidiscrete quadratic energy. The selected SSPRK2 witness has per-step
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
authenticated single-publisher helper. An installed Dim1 native receipt is
still required before claiming M23 runtime qualification.

Source-only checks: `test_coupled_gradient_hall_source.py` covers physical
identity, off-diagonal sign, provider selection, accepted-face publication,
component permutation and Hall zero. The independent review uses a separate
constant-matrix stencil/energy oracle, including three components. Neither
source generation nor that oracle is a native numerical result.
