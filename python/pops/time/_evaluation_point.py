"""Exact evaluation coordinates retained independently of split state endpoints."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from fractions import Fraction
from typing import Any

from pops.time.points import StagePoint, TimePoint


_ACTIVE_PARTITION: ContextVar[Any] = ContextVar("pops_evaluation_partition", default=None)
_EVALUATIONS = frozenset({
    "rhs", "diffusive_rhs", "source", "implicit_source", "local_transform", "apply",
    "solve_linear", "solve_local_linear", "solve_local_nonlinear", "solve_implicit_source",
})


@contextmanager
def evaluation_partition(program: Any, partition: str):
    """Scope a factory's physical subflow; only resulting evaluation IR retains the claim."""
    if not isinstance(partition, str) or not partition:
        raise ValueError("evaluation partition must be a non-empty string")
    token = _ACTIVE_PARTITION.set((program, partition))
    try:
        yield
    finally:
        _ACTIVE_PARTITION.reset(token)


def qualify_evaluation_attrs(program: Any, op: str, attrs: Any) -> Any:
    scope = _ACTIVE_PARTITION.get()
    if op not in _EVALUATIONS or scope is None or scope[0] is not program:
        return attrs
    return {**attrs, "evaluation_partition": scope[1]}


def evaluation_stage_fraction(value: Any, *, ark_partition: str | None = None) -> Fraction:
    """Select an authored partition, a common time, or an exact ARK operator coordinate.

    The ARK fallback is only for an operator whose semantics explicitly supplies its partition.
    A generic StagePoint with distinct coordinates never implies an explicit/implicit partition.
    """
    point = getattr(value, "point", None)
    partition = getattr(value, "attrs", {}).get("evaluation_partition")
    if partition is not None and (not isinstance(partition, str) or not partition):
        raise ValueError("evaluation_partition must be a non-empty string")
    if type(point) is TimePoint:
        selected = point
    elif type(point) is StagePoint:
        if partition is not None:
            if partition not in point.partitions:
                raise ValueError("evaluation partition %r is not declared by StagePoint %r"
                                 % (partition, point.name))
            selected = point.time_for(partition)
        else:
            try:
                selected = point.time
            except ValueError:
                if (ark_partition in {"explicit", "implicit"}
                        and set(point.partitions) == {"explicit", "implicit"}):
                    selected = point.time_for(ark_partition)
                else:
                    raise ValueError(
                        "evaluation at StagePoint %r has distinct partition times and requires "
                        "an explicit evaluation partition" % point.name) from None
    else:
        raise ValueError("evaluation requires an exact TimePoint or StagePoint")
    return Fraction(selected.step) + Fraction(selected.offset.to_python())


__all__ = ["evaluation_partition", "evaluation_stage_fraction", "qualify_evaluation_attrs"]
