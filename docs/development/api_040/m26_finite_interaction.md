# M26 finite symmetric interaction witness

The API 0.4.0 corpus closes a finite `N=12` witness with periodic
`W(x)=cos(2 pi x)`, positive density, explicit measure and pairing, and
`E(rho)=0.5 <rho,W*rho>`. The separate aggregation–diffusion PDE at
`N=32,64,128` has no prescribed diffusion coefficient or compatible
discrete gradient flux. This change makes no PDE, mass-evolution or energy
decay claim.

`FiniteMeasure` gives one strictly positive finite weight to each ordered
`FiniteSupport` degree of freedom. `FiniteSymmetricInteraction` requires a
finite, **exactly symmetric** represented kernel and constructs the existing
native `FiniteLinearMap` with entries `A_ij=W_ij*m_j`. Its action is
self-adjoint under `<a,b>_m=sum_i m_i a_i b_i`:

```text
<a,A b>_m = <A a,b>_m
E(rho) = 0.5 <rho,A rho>_m
DE(rho)[eta] = <eta,A rho>_m
E(rho+eta)-E(rho) = DE(rho)[eta] + 0.5 <eta,A eta>_m
```

Negative, zero, missing or nonfinite weights, a nonfinite kernel, and any
represented asymmetric pair are refused at authoring. Neither a tolerance
nor silent symmetrization alters the law. The generic finite classes do not
impose positivity on every possible input vector; the closed native witness
authenticates two positive finite density arrays before bind.

The public example `api040_m26_finite_interaction.py` represents the twelve
quadrature values as **components in one native cell**. They are explicit
finite degrees of freedom, not a flattened HPC mesh. It evaluates both
densities, the second by the measured adjoint, and saves the two native
potentials plus native values of both energies, a directional derivative,
both sides of the adjoint pairing, and both masses. One artifact is rebound
with a second pair of positive densities. A separate artifact permutes all
twelve DOFs while preserving physical identities and weights.

The independent oracle `api040_m26_finite_oracle.py` uses only the first
Fourier coefficient of each density; it does not call the production finite
map. It also cross-checks direct periodic quadrature in pure tests. Saved
native arrays are reopened and depermuted before comparison. The criteria,
fixed before native reception, are `2e-14` for unchanged densities,
`3e-13` for each native potential and measured scalar, `3e-13` for the
adjoint defect and exact quadratic energy increment, and `2e-14` for the
one-step clock. The one-cell step is a measurement transaction, not a
physical aggregation time step.

Source-only validation currently covers the generic authority checks,
independent Fourier identities, both orderings, two density binds and
complete Program C++ emission using the existing finite map kernel. These
checks do **not** authenticate the installed native library. The native/MPI
test is `tests/python/integration/runtime/test_api040_m26_finite_interaction_runtime.py`,
to be run by the repository's installed-check runner after root rebuilds
Dim1. A true spatial nonlocal convolution, gradient-compatible flux,
diffusion, conservation and full energy law remain separate obligations.
