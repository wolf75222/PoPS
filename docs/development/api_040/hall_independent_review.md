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

## Independent implementation review of 97f4c77c

The frozen author commit is `97f4c77c` in the separate
`PoPS-degenerate-diffusion` checkout. The reviewer changed no production code.
The additional public tests live in
`tests/python/unit/numerics/test_coupled_gradient_independent_review.py`.

The declaration retains the exact state owner and separate constant component
matrices D and R in its physical identity. A foreign state's identical names do
not authenticate it. RuntimeParam entries are explicitly rejected by this first
constant-matrix realization; they are not captured as frozen default values.
The numerical selection and native provider are distinct from TensorDiffusion.
The latter's component-Jacobian SPD test is bypassed only for the distinct
coupled provider, while finiteness and the spatial identity tensor remain
checked. No extra positive diagonal is introduced.

The inherited one-dimensional face formula reduces algebraically to
`(W[i+1]-W[i])/h`, with `W=(D+R)U`. Two additional independent NumPy tests use
three components, dense skew coupling, rank-deficient positive dissipation, and
a component permutation. They compare the gradient/Hessian face assembly with
the direct component-matrix Laplacian, and check zero skew energy and the
separate negative dissipative quadratic form. These are algebraic checks, not
an execution of the C++ kernel.

An actual source defect was caught before the author's freeze: the accepted-face
filter omitted the new `coupled_gradient` occurrence. Public validate/resolve/
Program emission produced zero `stage_accepted_exchanges` calls for SSPRK2.
The independent test failed with `0 == 2` (1 failed, 1.26 s). The corrected
97f4c77c produces both accepted stage exchanges. A separate test emits the
complete native state-storage loader, without a fabricated hyperbolic flux or
wave speed. The first coherent independent suite passed **17/17 in 2.10 s**.

Review also identified rank-local fixture hazards: conversion of root-owned
gathers on peers, root-only records returned to all-rank assertions, unchecked
bind/run/status exceptions before another collective, and root receipt writes
that could bypass a final broadcast. The frozen example converges these
operations, sends identical records to every rank, and its fixture checks files
only on rank 0 before broadcasting the verdict. This is a source review; it does
not prove termination of the native collectives themselves.

### PSD admission counterexample

An additional public test on 97f4c77c rejects the exact constant matrix
`D=ones((3,3))`, although `v.T D v=(sum(v))**2 >= 0` and it has two legitimate
null modes. The implementation tests the minimum numerical eigenvalue against
zero without an error model; on the review host `eigvalsh` returns a minimum
of approximately `-5.48266979e-16`. The public declaration raises
`ValueError: dissipative component matrix must be positive semidefinite`.
The paired negative test uses `diag(1,1,-2**-50)` and must continue to reject it;
simply adding a tolerance or projecting/clamping D is not an acceptable fix.
The discriminating run was **1 failed, 1 passed in 0.48 s**. This remains an
admission defect of the general declared PSD class, even though the M23 witness
uses D=0 and does not encounter it.

Follow-up `3fc51ec8dcb06094f5b12bcb6f430127f66fae77` replaces this eigensolver
decision by exact rational symmetric Schur complements of the represented
constants. A negative pivot refuses; a zero pivot requires its residual row to
vanish. No coefficient is changed, no small negative mode is admitted, and no
component-count limit is introduced. Inspection of the symmetric update and
replay of both adversaries confirms the correction.

Final independent result on `97f4c77c` + `3fc51ec8`: **19/19 passed in 4.92 s**
(seven public source/loader tests and twelve pure NumPy mathematical tests).
Ruff passes for both test files. These tests ran with `env -u PYTHONPATH` in
`pops-api040-c11`, with pytest `-o pythonpath` explicitly pointing to the frozen
author's Python source and absolute paths to the independent test files.
Verdict: favorable within this source and mathematical scope, with the two
initial defects retained above. No native build, JIT compilation, installed
run, GPU or AMR qualification was performed by this review; an authenticated
Dim1 installed run remains necessary for runtime qualification.

Follow-up `7ab9f572` removes an unnecessary coefficient-one/single-occurrence
restriction. Five further independent source cases cover weights `(1,2)`,
`(2,1)`, `(.5,1.5)`, `(.5,)`, and rejection of `(2,-.5)` despite its positive
total. Each SSPRK2 stage scales its shared RHS by the sum exactly once, while
its accepted face publication retains each occurrence ordinal and its own
physical coefficient separately from the temporal half-dt quadrature. The
tests verify distinct per-stage receivers and forbid duplicate ordinals; no
collapsed sum substitutes for the accepted exchange identities. Final replay
on 7ab9f572: **24/24 source/math tests passed in 5.35 s**, Ruff clean. Initial
test parsing assumed decimal half-dt and Real-wrapped float literals; it was
corrected to the emitter's exact rational half-dt and native floating literals,
without changing any production code or mathematical expectation.
