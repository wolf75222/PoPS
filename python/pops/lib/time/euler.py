"""pops.lib.time.euler -- Forward Euler time-stepping scheme.

Builds a pops.time.Program step for the classic first-order explicit method.
The backward_euler name is not defined here (the implicit BDF1 path is accessed
via bdf(..., order=1, linear_source=...)); only forward_euler lives here.
"""

from ._helpers import _stage_rate


def explicit_flow(P, state, frac=1.0, *, rhs_operator, fields_operator=None, name=None):
    """One explicit method-of-lines subflow from an existing State value.

    Returns ``state + frac * dt * R(state[, fields])`` without committing it. The rate and optional
    field solve are typed operator handles; no flux/source selectors are accepted.
    """
    R = _stage_rate(P, state, rhs_operator=rhs_operator, fields_operator=fields_operator,
                    tag=(name + "_") if name else "")
    return P.linear_combine(name, state + (float(frac) * P.dt) * R)


def forward_euler(P, block, *, rhs_operator, fields_operator=None):
    """Forward Euler: U^{n+1} = U + dt * R(U)."""
    U = P._state_value(block)
    P.commit(block, explicit_flow(P, U, 1.0, rhs_operator=rhs_operator,
                                  fields_operator=fields_operator, name="fe_step"))
