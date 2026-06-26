"""pops.lib.time.euler -- Forward Euler time-stepping scheme.

Builds a pops.time.Program step for the classic first-order explicit method.
The backward_euler name is not defined in pops.time (the implicit BDF1 path is
accessed via bdf(..., order=1, linear_source=...)); only forward_euler lives here.

# SPEC4-TODO: repoint to pops.time once it's a package.
"""

# _stage_rhs is a shared helper that lives in pops.time; import lazily to avoid
# a circular dependency during the Spec-4 transition period.


def _stage_rhs(P, U, sources, flux):
    # SPEC4-TODO: repoint to pops.lib.time._stage_rhs (or inline) once time.py is a package.
    from pops import time as _t  # noqa: PLC0415
    return _t._stage_rhs(P, U, sources, flux)


def forward_euler(P, block, *, sources=("default",), flux=True):
    """Forward Euler: U^{n+1} = U + dt * R(U)."""
    U = P.state(block)
    R = _stage_rhs(P, U, sources, flux)
    P.commit(block, P.linear_combine("fe_step", U + P.dt * R))
