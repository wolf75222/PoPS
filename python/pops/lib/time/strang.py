"""Generic sequential splitting as ordinary Program factories."""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from ._factory import instance_state, program_factory
from ._helpers import _stage_point


def _subflow(
    flow: Any,
    program: Any,
    state: Any,
    fraction: Any,
    point: Any,
    where: str,
) -> Any:
    """Author one exact sub-flow and authenticate its declared endpoint."""
    from pops.time.values import ProgramValue

    if not callable(flow):
        raise TypeError("%s must be a callable Program IR builder" % where)
    value = flow(program, state, fraction, at=point)
    if not isinstance(value, ProgramValue) or value.vtype != "state":
        raise TypeError("%s must return a Program state value" % where)
    if value.prog is not program:
        raise ValueError("%s returned a value owned by another Program" % where)
    if value.point != point:
        raise ValueError(
            "%s returned point %r instead of %r; materialize the sub-flow "
            "with program.value(..., at=at)" % (where, value.point, point)
        )
    return value


def _build_strang(program: Any, state: Any, first: Any, second: Any) -> None:
    temporal = instance_state(program, state, "Strang")
    initial = temporal.n
    after_first_half = _stage_point(
        program,
        "strang_first_half",
        partitions={"first": Fraction(1, 2), "second": 0},
    )
    after_second = _stage_point(
        program,
        "strang_second",
        partitions={"first": Fraction(1, 2), "second": 1},
    )
    stage = _subflow(
        first, program, initial, Fraction(1, 2), after_first_half, "Strang first[0]")
    stage = _subflow(second, program, stage, 1, after_second, "Strang second")
    endpoint = _subflow(
        first,
        program,
        stage,
        Fraction(1, 2),
        temporal.next.point,
        "Strang first[1]",
    )
    program.commit(temporal.next, endpoint)


def Strang(state: Any, *, first: Any, second: Any) -> Any:
    """Return ``first(dt/2) -> second(dt) -> first(dt/2)``.

    ``first`` and ``second`` are authoring-time callables with signature
    ``(program, state, fraction, *, at) -> ProgramValue``.  They build only
    public Program IR and must materialize their result at the supplied exact
    endpoint.  Partitioned stage coordinates retain both sub-flow clocks.
    """
    return program_factory("Strang", _build_strang, state, first, second)


def _build_lie(program: Any, state: Any, first: Any, second: Any) -> None:
    temporal = instance_state(program, state, "Lie")
    after_first = _stage_point(
        program,
        "lie_first",
        partitions={"first": 1, "second": 0},
    )
    stage = _subflow(first, program, temporal.n, 1, after_first, "Lie first")
    endpoint = _subflow(
        second, program, stage, 1, temporal.next.point, "Lie second")
    program.commit(temporal.next, endpoint)


def Lie(state: Any, *, first: Any, second: Any) -> Any:
    """Return the first-order sequential composition ``first(dt) -> second(dt)``."""
    return program_factory("Lie", _build_lie, state, first, second)


__all__ = ["Lie", "Strang"]
