# M23: independent mathematical preflight

This preflight audits the handoff's linear transverse Hall witness. It does not
qualify full Hall MHD or a native realization. The implementation review follows
after the author freezes the common reversible-gradient mechanism.

## Normative equation and a discriminating sign check

The preserved abstraction notes, lines 3067–3083, specify
`w_t = etaH J w_xx`, `J=[[0,-1],[1,0]]`. Thus the mode `cos(k*x)` has generator
`-etaH*k²*J`. With `k=2`, `etaH=.3`, `T=.2`, its signed rotation is **−.24**
radians, its frequency magnitude is **1.2**, and its Euclidean norm is constant.

The old handoff test `legacy/v0.2.0/tests/run_revision.py:260` instead exponentiates
`+etaH*k²*J`. Its norm-only assertion cannot detect the reversed rotation. The new
independent test preserves both versions as a discriminant: they have equal
norms but opposite measured phases and a component difference greater than .47.
The equation in the mathematical note is the authority for the new witness.

The oracle uses `[0,2*pi)` and exact cell averages of `(1,.2)*cos(2*x)`. Their
analytic sinc expression is separately checked by eight-point cell quadrature.
This domain fixes physical wavenumber 2 without confusing it with cycles per
unit interval. N=32/64/128 are mathematical test grids, not an implementation cap.

## Three distinct effects

- **Physical Hall term:** the skew component coupling conserves quadratic energy
  on the periodic domain. It must not be silently replaced by an SPD diffusion
  matrix, clamped, symmetrized, or given an artificial positive diagonal.
- **Spatial discretization:** the centered face-gradient/divergence pair has
  symbol `-lambda_h`, with `lambda_h=4*sin²(k*h/2)/h²`. It changes phase to
  `-etaH*lambda_h*T`, preserves semidiscrete norm, and has second-order phase
  error on these grids.
- **Time discretization:** for `y=etaH*lambda_h*dt`, Forward Euler amplifies by
  `sqrt(1+y²)` per step; SSPRK2 by `sqrt(1+y⁴/4)`. Backward Euler damps by
  `1/sqrt(1+y²)`. Midpoint conserves norm but rotates by `-2*atan(y/2)` per step.
  These are method effects, not Hall dissipation. Optional Ohmic diffusion
  multiplies the exact amplitude by `exp(-etaOhmic*k²*T)` and remains separately
  identified.

SSPRK2 has no nonzero stable interval on the imaginary axis. Choosing a small
step can make its growth measurable or negligible for a fixed smooth mode;
this is not a general stability certificate. In particular, `dt=C*h²` gives
highest-frequency total log amplification proportional to `dt³/h⁸`, hence
unbounded as h decreases at fixed C. A parabolic CFL number alone does not prove
uniform stability for this scheme. The test includes this distinction and the
finite imaginary-axis interval of the RK4 polynomial as a separate comparison.

## Evidence and remaining review

Files are new and use NumPy only: `api040_hall_independent_oracle.py` and
`test_hall_independent_oracle.py`. They import no PoPS expressions, lowering,
native generator or author's oracle. The tests exercise phase sign, exact FV
averages, Hall-zero identity, separate Ohmic damping, real periodic stencil,
spatial convergence and actual time-method amplification.

Source-only result: **10/10 passed** (0.19 s), using the dedicated
`pops-api040-c11` Python with `env -u PYTHONPATH`, pytest `-o pythonpath=python`.
Ruff passes for both new Python files without suppressed rules.

Initial direct sine-endpoint subtraction lost up to 1.42e−14 through division
by cell width at N128. The oracle was corrected to the equivalent stable sinc
formula, retaining the original 9e−15 comparison threshold; no native criterion
was weakened. This is a preflight implementation detail, not physical error.

No native/JIT build or installed reception was performed here. The common
mechanism still needs review of occurrence ownership, captures, component/axis
orientation, runtime domains, explicit stage authority and actual numerical
restriction. No claim about nonlinear Hall, AMR, full MHD, or multidimensional
physical Hall follows from this one-dimensional linear witness.
