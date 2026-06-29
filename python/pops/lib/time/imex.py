"""pops.lib.time.imex -- IMEX (implicit-explicit) time-stepping schemes.

Exports: imex_local, imex_local_linear.
"""

from ._helpers import _opcall


def imex_local(P, block, *, explicit_operator, implicit_operator, fields_operator=None,
               theta=1.0, state_space="U"):
    """IMEX with typed explicit-rate and implicit local-linear operator handles.

    This is the public ready-made IMEX macro. It is intentionally operator-first: it does not accept
    ``linear_source="..."``, ``sources=[...]`` or ``flux=True`` selectors. The model declares a typed
    explicit rate operator and a typed local-linear operator; this macro only composes them.
    """
    return imex_local_linear(P, block, explicit_operator=explicit_operator,
                             implicit_operator=implicit_operator,
                             fields_operator=fields_operator, theta=theta,
                             state_space=state_space)


def imex_local_linear(P, block, *, explicit_operator, implicit_operator, fields_operator=None,
                      theta=1.0, state_space="U"):
    """Generic IMEX with an explicit rate and an implicit local linear operator (Spec 2).

    One theta-implicit step of ``dU/dt = R(U[, fields]) + L([fields]) U``::

        U^{n+1} = (I - theta dt L)^{-1} (U^n + dt R)

    composing the typed ``explicit_operator`` and ``implicit_operator`` handles (and an optional
    ``fields_operator`` handle). Requires ``P.bind_operators(module)``.
    """
    if not (0.0 < theta <= 1.0):
        raise ValueError("imex_local_linear: theta must be in (0, 1]")
    u = P._state_value(block)
    fields = _opcall(P, fields_operator, u, value_name="fields") if fields_operator else None
    r = _opcall(P, explicit_operator, u, fields, value_name="R")
    lin = _opcall(P, implicit_operator, fields, value_name="L")
    q = P.linear_combine("imex_rhs", u + P.dt * r)
    u1 = P.solve_local_linear("imex_step", operator=P.I - theta * P.dt * lin, rhs=q, fields=fields)
    P.commit(block, u1)
    return u1
