"""Canonical Adams--Bashforth and local-linear BDF Program factories."""
from __future__ import annotations

from fractions import Fraction
from operator import index
from typing import Any

from pops.solvers import DenseLU
from pops.time import LocalLinear

from ._factory import (
    call_at, call_field_at, field_handle, instance_state, operator_handle,
    program_factory, resolve_solve_action,
)
from ._helpers import _block_label, _stage_point


_AB_WEIGHTS = {
    1: (1,),
    2: (Fraction(3, 2), Fraction(-1, 2)),
    3: (Fraction(23, 12), Fraction(-16, 12), Fraction(5, 12)),
}


def _validated_order(order: Any, supported: Any, scheme: str) -> int:
    """Return an exact integer order before it is used in names or graph construction."""
    if isinstance(order, bool):
        raise ValueError("%s order must be %s" % (scheme, supported))
    try:
        normalized = index(order)
    except TypeError:
        raise ValueError("%s order must be %s" % (scheme, supported)) from None
    if normalized not in supported:
        raise ValueError("%s order must be %s" % (scheme, supported))
    return int(normalized)


def _history(program: Any, temporal: Any, name: str, lag: int, space: Any) -> Any:
    return program.history(
        name,
        lag=lag,
        space=space,
        block=temporal.block,
        state_ref=temporal.state,
    )


def _build_adams_bashforth(
    program: Any,
    state: Any,
    rate: Any,
    fields: Any,
    order: int,
    solve_action: Any,
) -> None:
    order = _validated_order(order, (1, 2, 3), "AdamsBashforth")
    rate = operator_handle(rate, "AdamsBashforth rate")
    if fields is not None:
        fields = field_handle(fields, "AdamsBashforth fields")
    temporal = instance_state(program, state, "AdamsBashforth")
    initial = temporal.n
    point = _stage_point(program, "ab%d_current" % order, 0)
    stage_fields = call_field_at(
        program, fields, initial, name="ab%d_fields" % order, point=point,
        solve_action=solve_action,
    ) if fields is not None else None
    current = call_at(
        program, rate, initial, stage_fields,
        name="ab%d_rate" % order, point=point,
    )
    expression = initial + program.dt * _AB_WEIGHTS[order][0] * current
    if order > 1:
        history_name = _block_label(temporal) + ".rate"
        program.store_history(history_name, current)
        for lag, coefficient in enumerate(_AB_WEIGHTS[order][1:], start=1):
            previous = _history(
                program, temporal, history_name, lag, current.space)
            expression = expression + program.dt * coefficient * previous
    endpoint = program.value(
        "ab%d_step" % order, expression, at=temporal.next.point)
    program.commit(temporal.next, endpoint)


def AdamsBashforth(
    state: Any,
    *,
    rate: Any,
    order: int,
    fields: Any = None,
    solve_action: Any = None,
) -> Any:
    """Return an ordinary explicit multistep Program with typed operator dependencies."""
    order = _validated_order(order, (1, 2, 3), "AdamsBashforth")
    action = resolve_solve_action(solve_action, "AdamsBashforth")
    return program_factory(
        "AdamsBashforth%d" % order,
        _build_adams_bashforth,
        state,
        rate,
        fields,
        order,
        action,
    )


def _build_bdf(
    program: Any,
    state: Any,
    implicit: Any,
    explicit: Any,
    fields: Any,
    order: int,
    solve_action: Any,
) -> None:
    order = _validated_order(order, (1, 2), "BDF")
    implicit = operator_handle(implicit, "BDF implicit")
    if explicit is not None:
        explicit = operator_handle(explicit, "BDF explicit")
    if fields is not None:
        fields = field_handle(fields, "BDF fields")
    temporal = instance_state(program, state, "BDF")
    initial = temporal.n
    point = _stage_point(
        program,
        "bdf%d_stage" % order,
        partitions={"explicit": 0, "implicit": 1},
    )
    stage_fields = call_field_at(
        program, fields, initial, name="bdf%d_fields" % order, point=point,
        solve_action=solve_action,
    ) if fields is not None else None
    linear = call_at(
        program, implicit, stage_fields,
        name="bdf%d_linear" % order, point=point,
    )
    explicit_rate = call_at(
        program, explicit, initial, stage_fields,
        name="bdf%d_explicit" % order, point=point,
    ) if explicit is not None else None
    if order == 1:
        rhs_expression = initial
        if explicit_rate is not None:
            rhs_expression = rhs_expression + program.dt * explicit_rate
        operator = program.I - program.dt * linear
    else:
        history_name = _block_label(temporal) + ".state"
        program.store_history(history_name, initial)
        previous = _history(
            program, temporal, history_name, 1, initial.space)
        rhs_expression = Fraction(4, 3) * initial - Fraction(1, 3) * previous
        if explicit_rate is not None:
            rhs_expression = rhs_expression + Fraction(2, 3) * program.dt * explicit_rate
        operator = program.I - Fraction(2, 3) * program.dt * linear
    rhs = program.value("bdf%d_rhs" % order, rhs_expression, at=point)
    solved = program.solve(
        LocalLinear(operator=operator, rhs=rhs, fields=stage_fields),
        solver=DenseLU(), name="bdf%d_solve" % order,
    ).consume(action=solve_action)
    endpoint = program.value(
        "bdf%d_step" % order, solved, at=temporal.next.point)
    program.commit(temporal.next, endpoint)


def BDF(
    state: Any,
    *,
    implicit: Any,
    order: int,
    explicit: Any = None,
    fields: Any = None,
    solve_action: Any = None,
) -> Any:
    """Return BDF1/BDF2 for one typed local-linear implicit operator.

    Globally coupled nonlinear problems remain explicit ``Program.solve(problem, solver=...)``
    authoring; this preset does not guess a Jacobian, preconditioner, or field policy.
    """
    order = _validated_order(order, (1, 2), "BDF")
    action = resolve_solve_action(solve_action, "BDF")
    return program_factory(
        "BDF%d" % order, _build_bdf,
        state, implicit, explicit, fields, order, action,
    )


__all__ = ["AdamsBashforth", "BDF"]
