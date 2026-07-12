"""pops.lib.time.predictor_corrector -- Predictor-corrector scheme (operator-first, Spec 2).

Exports: predictor_corrector_local_linear.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from ._helpers import (
    _at_point, _commit, _opcall, _operator_handle, _stage_point, _time_state, program_macro,
)


@program_macro
def predictor_corrector_local_linear(P: Any, block: Any, state: Any = None, *,
                                     fields_operator: Any,
                                     explicit_rate_operator: Any, implicit_operator: Any,
                                     commit: Any = True) -> Any:
    """Generic predictor-corrector for ``dU/dt = R(U, fields) + L(fields) U`` (Spec 2, operator-first).

    Composes THREE typed operators by name -- a field operator ``fields_operator: U -> Fields``, an
    explicit rate ``explicit_rate_operator: (U, Fields) -> Rate(U)`` and a local linear operator
    ``implicit_operator: Fields -> LocalLinearOperator(U, U)`` -- into one trapezoidal step with the
    L term treated implicitly via local solves::

        U*    = (I - dt L_n)^{-1} (U^n + dt R_n)
        U^n+1 = (I - 1/2 dt L*)^{-1} (U^n + 1/2 dt R_n + 1/2 dt R* + 1/2 dt L* U*)

    It mentions no physics; ``state_space`` is informational. Each of ``fields_operator`` /
    ``explicit_rate_operator`` / ``implicit_operator`` is a typed :class:`pops.model.OperatorHandle`
    from an ``m.*`` declarer, not a name string. Requires ``P.bind_operators(module)``.
    """
    fields_operator = _operator_handle(fields_operator, "fields_operator")
    explicit_rate_operator = _operator_handle(explicit_rate_operator, "explicit_rate_operator")
    implicit_operator = _operator_handle(implicit_operator, "implicit_operator")
    temporal = _time_state(P, block, state)
    u_n = temporal.n
    predictor = _stage_point(
        P, "predictor", partitions={"explicit": 0, "implicit": 1})
    fields_n = _opcall(P, fields_operator, u_n, value_name="fields_n", point=predictor)
    r_n = _opcall(
        P, explicit_rate_operator, u_n, fields_n, value_name="R_n", point=predictor)
    l_n = _opcall(P, implicit_operator, fields_n, value_name="L_n", point=predictor)
    u_star = _at_point(P, P.solve_local_linear(
        "U_star", operator=P.I - P.dt * l_n,
        rhs=P.linear_combine("U_star_rhs", u_n + P.dt * r_n, at=predictor),
        fields=fields_n), predictor)
    corrector = _stage_point(
        P, "corrector", partitions={"explicit": 1, "implicit": 1})
    fields_star = _opcall(
        P, fields_operator, u_star, value_name="fields_star", point=corrector)
    r_star = _opcall(
        P, explicit_rate_operator, u_star, fields_star, value_name="R_star", point=corrector)
    l_star = _opcall(
        P, implicit_operator, fields_star, value_name="L_star", point=corrector)
    c_star = _at_point(
        P, P.apply(l_star, u_star, fields=fields_star, name="C_star"), corrector)
    half = Fraction(1, 2)
    q = P.linear_combine(
        "Q", u_n + half * P.dt * r_n + half * P.dt * r_star + half * P.dt * c_star,
        at=temporal.next.point)
    u_np1 = P.solve_local_linear(
        "U_np1", operator=P.I - half * P.dt * l_star, rhs=q, fields=fields_star)
    if commit:
        _commit(P, temporal, u_np1)
    return u_np1
