"""Canonical explicit Runge--Kutta Program factories."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from pops.time._methods.tableau import RungeKuttaTableau

from ._factory import (
    _call_field_at,
    call_at,
    exact_ordered_routes,
    field_handle,
    instance_state,
    operator_handle,
    program_factory,
    resolve_solve_action,
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


@dataclass(frozen=True, slots=True)
class RungeKuttaRoute:
    """One exact block-state to owner-qualified rate route in an explicit RK system."""

    state: Any
    rate: Any

    def __post_init__(self) -> None:
        from pops.model import OperatorHandle, StateHandle

        if not isinstance(self.state, StateHandle) or not self.state.is_instance:
            raise TypeError(
                "RungeKuttaRoute state must be the exact block-qualified StateHandle "
                "produced by block[state]"
            )
        if not isinstance(self.rate, OperatorHandle):
            raise TypeError("RungeKuttaRoute rate must be an exact owner-qualified OperatorHandle")
        declaration = self.rate.declaration_ref if self.rate.is_instance else self.rate
        state_declaration = self.state.declaration_ref
        if (
            declaration is None
            or state_declaration is None
            or declaration.owner_path != state_declaration.owner_path
        ):
            raise ValueError("RungeKuttaRoute state and rate must belong to the same model owner")
        if self.rate.is_instance and self.rate.block_ref != self.state.block_ref:
            raise ValueError(
                "RungeKuttaRoute instance rate and state must belong to the same block"
            )


def _route_name(base: str, route: int, count: int) -> str:
    return base if count == 1 else "%s_route_%d" % (base, route)


def _build_explicit_runge_kutta(
    program: Any,
    routes: tuple[RungeKuttaRoute, ...],
    fields: Any,
    tableau: Any,
    solve_action: Any,
) -> None:
    if type(tableau) is not RungeKuttaTableau:
        raise TypeError("RungeKutta tableau must be an exact RungeKuttaTableau")
    routes = exact_ordered_routes(routes, RungeKuttaRoute, "RungeKutta")
    rates_by_route = tuple(
        operator_handle(route.rate, "RungeKutta route %d rate" % index)
        for index, route in enumerate(routes)
    )
    if fields is not None:
        fields = field_handle(fields, "RungeKutta fields")
        blocks = tuple(route.state.block_ref for route in routes)
        if len(set(blocks)) != len(blocks):
            raise ValueError(
                "RungeKutta shared fields are ambiguous for multiple state routes on one block"
            )
    temporals = tuple(
        instance_state(program, route.state, "RungeKutta route %d" % index)
        for index, route in enumerate(routes)
    )
    initials = tuple(temporal.n for temporal in temporals)
    from pops.model import operator_family
    from pops.time.operator_resolution import resolve_operator_handle

    for index, (rate, initial) in enumerate(zip(rates_by_route, initials, strict=True)):
        operator = resolve_operator_handle(
            program,
            rate,
            where="RungeKutta route %d rate" % index,
            values=(initial,),
        )
        if operator_family(operator.kind) != "rate":
            raise ValueError(
                "RungeKutta route %d operator %r has registered kind %r, not a rate family"
                % (index, operator.name, operator.kind)
            )
    stage_rates: list[list[Any]] = [[] for _ in routes]
    tag = tableau.name or "runge_kutta"
    for stage in range(tableau.stages):
        point = _stage_point(program, "%s_stage_%d" % (tag, stage), tableau.c[stage])
        stage_states = []
        for route_index, initial in enumerate(initials):
            stage_state = initial
            if stage:
                expression = initial
                for previous, coefficient in enumerate(tableau.A[stage]):
                    if coefficient != 0:
                        expression = (
                            expression
                            + program.dt * coefficient * stage_rates[route_index][previous]
                        )
                stage_state = program.value(
                    _route_name("%s_U%d" % (tag, stage), route_index, len(routes)),
                    expression,
                    at=point,
                )
            stage_states.append(stage_state)
        field_call = (
            _call_field_at(
                program,
                fields,
                *stage_states,
                name="%s_fields_%d" % (tag, stage),
                point=point,
                solve_action=solve_action,
            )
            if fields is not None
            else None
        )
        rate_states = tuple(stage_states) if field_call is None else field_call.states
        stage_fields = None if field_call is None else field_call.fields
        for route_index, (rate, rate_state) in enumerate(
            zip(rates_by_route, rate_states, strict=True)
        ):
            stage_rates[route_index].append(
                call_at(
                    program,
                    rate,
                    rate_state,
                    stage_fields,
                    name=_route_name("%s_k_%d" % (tag, stage), route_index, len(routes)),
                    point=point,
                )
            )
    endpoints = {}
    for route_index, (temporal, initial, rates) in enumerate(
        zip(temporals, initials, stage_rates, strict=True)
    ):
        result = initial
        for coefficient, stage_rate in zip(tableau.b, rates, strict=True):
            if coefficient != 0:
                result = result + program.dt * coefficient * stage_rate
        endpoint = program.value(
            _route_name("%s_step" % tag, route_index, len(routes)),
            result,
            at=temporal.next.point,
        )
        endpoints[temporal.next] = endpoint
    program.commit_many(endpoints)


def _runge_kutta(
    state: Any = None,
    *,
    rate: Any = None,
    routes: tuple[RungeKuttaRoute, ...] | None = None,
    tableau: RungeKuttaTableau,
    fields: Any = None,
    solve_action: Any = None,
    program_name: str | None = None,
    where: str = "RungeKutta",
) -> Any:
    """Build one explicit-RK Program after normalizing the public route forms."""
    if routes is None:
        if state is None or rate is None:
            raise TypeError(
                "RungeKutta requires both state= and rate=, or routes=(RungeKuttaRoute(...),)"
            )
        normalized = (RungeKuttaRoute(state, rate),)
    else:
        if state is not None or rate is not None:
            raise TypeError(
                "RungeKutta routes= is mutually exclusive with the single-state state=/rate= form"
            )
        normalized = exact_ordered_routes(routes, RungeKuttaRoute, "RungeKutta")
    name = program_name
    if name is None:
        name = tableau.name if type(tableau) is RungeKuttaTableau and tableau.name else "RungeKutta"
    action = resolve_solve_action(solve_action, where)
    return program_factory(name, _build_explicit_runge_kutta, normalized, fields, tableau, action)


def RungeKutta(
    state: Any = None,
    *,
    rate: Any = None,
    routes: tuple[RungeKuttaRoute, ...] | None = None,
    tableau: RungeKuttaTableau,
    fields: Any = None,
    solve_action: Any = None,
) -> Any:
    """Return an explicit-RK Program for one route or an exact ordered route tuple."""
    return _runge_kutta(
        state,
        rate=rate,
        routes=routes,
        tableau=tableau,
        fields=fields,
        solve_action=solve_action,
    )


def RK4(state: Any, *, rate: Any, fields: Any = None, solve_action: Any = None) -> Any:
    """Return the classic fourth-order Runge--Kutta Program."""
    return _runge_kutta(
        state,
        rate=rate,
        fields=fields,
        tableau=RK4_TABLEAU,
        solve_action=solve_action,
        program_name="RK4",
        where="RK4",
    )


__all__ = [
    "ButcherTableau",
    "RK4",
    "RK4_TABLEAU",
    "RungeKutta",
    "RungeKuttaRoute",
    "SSPRK2_TABLEAU",
]
