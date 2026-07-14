"""Canonical explicit Runge--Kutta Program factories."""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from pops.time._methods.tableau import RungeKuttaTableau

from ._factory import (
    call_at, call_field_at, field_handle, instance_state, operator_handle,
    program_factory, resolve_solve_action,
)
from ._helpers import _stage_point


ButcherTableau = RungeKuttaTableau


RK4_TABLEAU = ButcherTableau(
    A=[[], [Fraction(1, 2)], [0, Fraction(1, 2)], [0, 0, 1]],
    b=[Fraction(1, 6), Fraction(1, 3), Fraction(1, 3), Fraction(1, 6)],
    c=[0, Fraction(1, 2), Fraction(1, 2), 1],
    name="rk4",
)

SSPRK2_TABLEAU = ButcherTableau(
    A=[[], [1]],
    b=[Fraction(1, 2), Fraction(1, 2)],
    c=[0, 1],
    name="ssprk2",
)


def _build_explicit_runge_kutta(
    program: Any,
    state: Any,
    rate: Any,
    fields: Any,
    tableau: Any,
    solve_action: Any,
) -> None:
    if type(tableau) is not RungeKuttaTableau:
        raise TypeError("RungeKutta tableau must be an exact RungeKuttaTableau")
    rate = operator_handle(rate, "RungeKutta rate")
    if fields is not None:
        fields = field_handle(fields, "RungeKutta fields")
    temporal = instance_state(program, state, "RungeKutta")
    initial = temporal.n
    rates = []
    tag = tableau.name or "runge_kutta"
    for stage in range(tableau.stages):
        point = _stage_point(
            program, "%s_stage_%d" % (tag, stage), tableau.c[stage])
        stage_state = initial
        if stage:
            expression = initial
            for previous, coefficient in enumerate(tableau.A[stage]):
                if coefficient != 0:
                    expression = expression + program.dt * coefficient * rates[previous]
            stage_state = program.value(
                "%s_U%d" % (tag, stage), expression, at=point)
        stage_fields = call_field_at(
            program, fields, stage_state,
            name="%s_fields_%d" % (tag, stage), point=point,
            solve_action=solve_action,
        ) if fields is not None else None
        rates.append(call_at(
            program, rate, stage_state, stage_fields,
            name="%s_k_%d" % (tag, stage), point=point,
        ))
    result = initial
    for coefficient, stage_rate in zip(tableau.b, rates, strict=True):
        if coefficient != 0:
            result = result + program.dt * coefficient * stage_rate
    endpoint = program.value(
        "%s_step" % tag, result, at=temporal.next.point)
    program.commit(temporal.next, endpoint)


def RungeKutta(
    state: Any,
    *,
    rate: Any,
    tableau: RungeKuttaTableau,
    fields: Any = None,
    solve_action: Any = None,
) -> Any:
    """Return an ordinary explicit-RK Program for one exact ``block[state]`` handle."""
    name = tableau.name if type(tableau) is RungeKuttaTableau and tableau.name else "RungeKutta"
    action = resolve_solve_action(solve_action, "RungeKutta")
    return program_factory(
        name, _build_explicit_runge_kutta, state, rate, fields, tableau, action)


def RK4(state: Any, *, rate: Any, fields: Any = None, solve_action: Any = None) -> Any:
    """Return the classic fourth-order Runge--Kutta Program."""
    action = resolve_solve_action(solve_action, "RK4")
    return program_factory(
        "RK4", _build_explicit_runge_kutta, state, rate, fields, RK4_TABLEAU, action)


__all__ = [
    "ButcherTableau", "RK4", "RK4_TABLEAU", "RungeKutta", "SSPRK2_TABLEAU",
]
