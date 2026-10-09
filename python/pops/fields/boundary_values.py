"""Explicit pointwise values available to compiled boundary providers.

These nodes keep declaration identity separate from symbolic algebra.  A Handle remains a
Boolean/hashable identity; callers opt into a pointwise boundary read and, for vector states,
must name the component they intend to consume.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pops._ir.expr import Const, Expr
from pops.model import Handle


# Compatibility import paths retain the exact common expression identity.
from pops.model.pointwise_boundary import BoundaryValue, _component_index


class LogicalTimeCoordinate(Enum):
    TIME = "time"
    DT = "dt"
    STEP = "step"
    SUBSTEP = "substep"
    ITERATION = "iteration"
    STAGE = "stage"
    PARTITION = "partition"


class LogicalTimeValue(Expr):
    """One exact value from the Program-supplied ``FieldLogicalTimePoint``."""

    __slots__ = ("coordinate",)

    def __init__(self, coordinate: LogicalTimeCoordinate | str) -> None:
        try:
            coordinate = (coordinate if isinstance(coordinate, LogicalTimeCoordinate)
                          else LogicalTimeCoordinate(coordinate))
        except (TypeError, ValueError):
            raise ValueError(
                "logical_time coordinate must be one of %s"
                % [item.value for item in LogicalTimeCoordinate]) from None
        self.coordinate = coordinate.value

    def __pops_ir_children__(self) -> tuple:
        return ()

    def __pops_ir_key__(self, recurse: Any) -> Any:
        return ("field_logical_time", self.coordinate)

    def __pops_ir_diff__(self, *, recurse: Any, target: Any, definitions: Any) -> Expr:
        return Const(0)

    def eval(self, env: Any) -> Any:
        key = "pops.field.logical_time.%s" % self.coordinate
        if key not in env:
            raise KeyError("logical time value %r missing from the environment" % key)
        return env[key]

    def deps(self) -> set[str]:
        return {"pops.field.logical_time.%s" % self.coordinate}

    def _str(self) -> str:
        return "logical_time(%r)" % self.coordinate


def boundary_value(handle: Any, component: Any = None) -> BoundaryValue:
    return BoundaryValue(handle, component)


def logical_time(coordinate: LogicalTimeCoordinate | str = "time") -> LogicalTimeValue:
    return LogicalTimeValue(coordinate)


__all__ = [
    "BoundaryValue", "LogicalTimeCoordinate", "LogicalTimeValue", "boundary_value",
    "logical_time",
]
