"""pops.lib.time.euler -- Forward Euler time-stepping scheme.

Builds a pops.time.Program step for the classic first-order explicit method.
The backward_euler name is not defined here (the implicit BDF1 path is accessed
via bdf(..., order=1, linear_source=...)); only forward_euler lives here.
"""

from ._helpers import _stage_rhs


def explicit_flow(P, state, frac=1.0, *, sources=("default",), flux=True, name=None):
    """One explicit method-of-lines subflow from an existing State value.

    Returns ``state + frac * dt * R(state)`` without committing it. Splitting macros use this helper
    for half-steps so examples and users do not call Program-private RHS builders directly.
    """
    R = _stage_rhs(P, state, sources, flux)
    return P.linear_combine(name, state + (float(frac) * P.dt) * R)


def forward_euler(P, block, *, sources=("default",), flux=True):
    """Forward Euler: U^{n+1} = U + dt * R(U)."""
    U = P.state(block)
    P.commit(block, explicit_flow(P, U, 1.0, sources=sources, flux=flux, name="fe_step"))
