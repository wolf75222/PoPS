# Computed frontier proposal clipping

`ComputedDt(dt, clip_to_frontier=True)` requests `min(dt, limit - start)` at each
accepted entry. A rejected attempt shrinks that capped proposal. The Program
still computes the accepted duration; clipping does not authorize an overshoot
or replace the returned duration. Existing endpoint ULP bounds are unchanged.

This explicit mode has a version 2 strategy descriptor with
`request_limit="remaining"`. The default `clip_to_frontier=False` preserves the
version 1 nominal request and its exact serialized representation, including
one-step methods which request 1 and accept duration 0.8.

A strict restart reconstructs the first request from the stored start and run
limit, then applies the exact recorded rejection count. A changed limit,
request, policy, or retry count cannot silently authorize a different step.
The collective preparation identity already includes the complete strategy,
entry, limit, and preceding receipt.

The T10 default corpus selection opts into clipping for a finite requested run
endpoint. Its separate one-step affine finalization receivers retain nominal
requests. The multipas reference independently evaluates the two-level Heun
rotation in NumPy, using only patch volumes and the authored step policy; it
checks the rotation trajectory, the chosen quadratic metric, and passive mass.
Native reception of this addition is pending; source checks do not qualify it.


`Program.composite_integral(..., over=state, level=p.int64(k))` selects one
level of the immutable composite tower. The default omits `level` and integrates
all levels. A selected level still intersects the finest active support, including
coverage and cut-cell fractions. The native invocation validates the range and
agrees on the exact domain across MPI ranks before reducing any integrand.

The T10 rotation witness explicitly integrates the radial increment and squared
increment of its authored Heun polynomial: `R(h)-1` on the coarse level and
`R(h/2)^2-1` for two fine substeps. This computes the coefficient before adding
the increment to the state. Subtracting already rounded endpoint states loses
the small radial component near the run frontier. These equations belong to the
Python example; the engine supplies only the selected physical integration domain
and the existing affine publication contract. No factor is clamped or invented.
