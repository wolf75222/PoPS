# M10 periodic self-consistent fitted flux witness

The existing Scharfetter–Gummel route already binds one declared drift
occurrence and one diffusion occurrence to a shared fitted face flux. Its
existing solved-field test uses a fixed charge block independent of the
transported density. The new public witness in
`tests/python/integration/runtime/test_m10_self_consistent_sg.py` makes the
Poisson load depend on the **current density** at the same stage.

This is the explicitly bounded periodic, neutral-background variant. On
`N=32` cell centers let `psi=0.4 cos(2 pi x)`, `n*=exp(-psi)`, and
`b=n* - (-Delta_h psi)`, where `-Delta_h` is the actual centered periodic
Poisson operator. A fixed background block stores `b`. The field equation is
`-Delta_h phi=n-b`, solved at stage `c=0` from the live density and background,
then published to the density's exact provider before its fitted rate.
The program accepts one Forward Euler step and stores the field observation.

The perturbed initial density is `n*+0.08 cos(4 pi x)`. An independent
discrete eigenvalue calculation gives the stage potential
`psi+[0.08/(4 N^2 sin^2(2 pi/N))] cos(4 pi x)`. The face oracle evaluates
`D/h [B(-delta)n_R-B(delta)n_L]` with scalar `expm1`/series Bernoulli,
then the oriented conservative divergence. It checks the accepted state,
latest published potential (history slot 1 after commit), exact signed joint
occurrence coverage and the per-cell accepted exchange ledger. The same
one-step oracle with the old potential differs by more than `1e-8`, making a
stale field observable. The unperturbed pair has zero SG flux to roundoff.

This test does not claim physical-wall boundary closure, transient
convergence, positivity for arbitrary data, or entropy dissipation. Those
are separate parts of the M10 matrix. Source validation/resolution/emission
and the independent finite-difference oracle pass locally; native execution
awaits the central rebuilt package and a receipt on its exact identity.
