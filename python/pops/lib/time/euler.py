"""pops.lib.time.euler -- Forward Euler time-stepping scheme.

Builds a pops.time.Program step for the classic first-order explicit method.
The backward_euler name is not defined here (the implicit BDF1 path is accessed
via bdf(..., order=1, linear_source=...)); only forward_euler lives here.
"""

from ._helpers import _stage_rhs, program_macro


@program_macro
def forward_euler(P, block, *, sources=("default",), flux=True):
    """Forward Euler: U^{n+1} = U + dt * R(U).

    Called with a live ``Program`` first (``forward_euler(P, block)``) it mutates ``P`` in place; called
    with the block name (``forward_euler(block)``) it returns a fresh, inspectable Program (ADC-554)."""
    U = P.state(block)
    R = _stage_rhs(P, U, sources, flux)
    P.commit(block, P.linear_combine("fe_step", U + P.dt * R))
