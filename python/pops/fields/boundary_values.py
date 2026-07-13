"""Explicit pointwise values available to compiled boundary providers.

These nodes keep declaration identity separate from symbolic algebra.  A Handle remains a
Boolean/hashable identity; callers opt into a pointwise boundary read and, for vector states,
must name the component they intend to consume.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pops.ir.expr import Const, Expr
from pops.model import Handle


def _component_index(handle: Handle, component: Any) -> tuple[int, str | None]:
    if handle.kind == "field":
        if component not in (None, 0):
            raise ValueError("a scalar field boundary read accepts only component 0")
        return 0, None
    if handle.kind != "state":
        raise TypeError("boundary_value requires a state or field Handle")
    declaration = handle.declaration_ref or handle
    space = getattr(declaration, "space", None)
    components = tuple(getattr(space, "components", ()))
    if not components:
        components = tuple(getattr(declaration, "components", ()))
    if not components:
        raise TypeError(
            "state boundary reads require authoritative component metadata on the Handle")
    if isinstance(component, str):
        try:
            return components.index(component), component
        except ValueError:
            raise KeyError(
                "state %r has no component %r (have: %s)"
                % (handle.local_id, component, ", ".join(components))) from None
    if isinstance(component, bool) or not isinstance(component, int):
        raise TypeError("a state boundary read requires a component name or integer index")
    if component < 0 or component >= len(components):
        raise IndexError("state boundary component index %d is out of range" % component)
    return component, components[component]


class BoundaryValue(Expr):
    """Pointwise read of one resolved state component or scalar field."""

    __slots__ = ("handle", "component", "component_name")

    def __init__(self, handle: Any, component: Any = None) -> None:
        if not isinstance(handle, Handle):
            raise TypeError("BoundaryValue requires a declaration Handle")
        index, name = _component_index(handle, component)
        self.handle = handle
        self.component = index
        self.component_name = name

    def resolve_references(self, resolver: Any) -> BoundaryValue:
        from ._references import resolve_handle

        resolved = resolve_handle(self.handle, resolver, where="BoundaryValue handle")
        return BoundaryValue(resolved, self.component)

    def declaration_references(self) -> tuple[Handle, ...]:
        return (self.handle,)

    def __pops_ir_children__(self) -> tuple:
        return ()

    def __pops_ir_key__(self, recurse: Any) -> Any:
        return ("boundary_value", self.handle.qualified_id, self.component)

    def __pops_ir_diff__(self, *, recurse: Any, target: Any, definitions: Any) -> Expr:
        target_handle = getattr(target, "handle", target)
        same = isinstance(target_handle, Handle) and target_handle == self.handle
        return Const(1 if same and self.component == 0 else 0)

    def eval(self, env: Any) -> Any:
        key = (self.handle.qualified_id, self.component)
        if key not in env:
            raise KeyError("boundary value %r missing from the environment" % (key,))
        return env[key]

    def deps(self) -> set[Any]:
        return {(self.handle.qualified_id, self.component)}

    def _str(self) -> str:
        suffix = self.component_name if self.component_name is not None else self.component
        return "boundary_value(%s, %r)" % (self.handle.qualified_id, suffix)


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
