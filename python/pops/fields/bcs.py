"""Extensible typed boundary selectors and field-value conditions."""

from __future__ import annotations

from typing import Any

from pops.descriptors import Descriptor
from pops.descriptors_report import CapabilitySet
from pops.ir.expr import Expr
from pops.model import Handle

from ._identity import strict_field_data
from ._references import collect_references, reference_label, resolve_handle, resolve_value


class BoundarySelector(Descriptor):
    """Small extension interface selecting a topological boundary region."""

    category = "boundary_selector"

    def to_data(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "options": self.options()}


class AllPhysicalBoundaries(BoundarySelector):
    def options(self) -> dict[str, Any]:
        return {"selector": "all_physical"}


class AxisBoundary(BoundarySelector):
    """Cartesian boundary selector usable in any dimension."""

    def __init__(self, axis: Any, side: Any) -> None:
        if isinstance(axis, bool) or not isinstance(axis, int) or axis < 0:
            raise ValueError("AxisBoundary axis must be an integer >= 0")
        if side not in {"lo", "hi"}:
            raise ValueError("AxisBoundary side must be exactly 'lo' or 'hi'")
        self.axis = axis
        self.side = side

    def options(self) -> dict[str, Any]:
        return {"selector": "axis", "axis": self.axis, "side": self.side}


class NamedBoundary(BoundarySelector):
    """Reference a model/domain-owned boundary for non-Cartesian geometries."""

    def __init__(self, boundary: Any) -> None:
        if not isinstance(boundary, Handle):
            raise TypeError("NamedBoundary boundary must be a declaration Handle")
        self.boundary = boundary

    def options(self) -> dict[str, Any]:
        return {
            "selector": "named",
            "boundary": reference_label(self.boundary, where="NamedBoundary boundary"),
        }

    def resolve_references(self, resolver: Any) -> NamedBoundary:
        return NamedBoundary(
            resolve_handle(self.boundary, resolver, where="NamedBoundary boundary")
        )

    def declaration_references(self) -> tuple[Handle, ...]:
        return (self.boundary,)


class XMin(AxisBoundary):
    def __init__(self) -> None:
        super().__init__(0, "lo")


class XMax(AxisBoundary):
    def __init__(self) -> None:
        super().__init__(0, "hi")


class YMin(AxisBoundary):
    def __init__(self) -> None:
        super().__init__(1, "lo")


class YMax(AxisBoundary):
    def __init__(self) -> None:
        super().__init__(1, "hi")


class ZMin(AxisBoundary):
    def __init__(self) -> None:
        super().__init__(2, "lo")


class ZMax(AxisBoundary):
    def __init__(self) -> None:
        super().__init__(2, "hi")


class FieldBoundaryCondition(Descriptor):
    """Small extension interface for one field boundary law."""

    category = "field_bc"

    def to_data(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "options": self.options()}


def _boundary_value(value: Any) -> Any:
    if isinstance(value, (Expr, Handle)):
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if callable(getattr(value, "to_data", None)):
        return value
    raise TypeError("boundary value must be a scalar, Handle, Expr, or typed value with to_data()")


class Periodic(FieldBoundaryCondition):
    """Require consistency with a periodic mesh-topology pairing."""

    def options(self) -> dict[str, Any]:
        return {"bc": "periodic"}

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet({"periodic": True})


class Dirichlet(FieldBoundaryCondition):
    def __init__(self, value: Any = 0) -> None:
        self.value = _boundary_value(value)

    def options(self) -> dict[str, Any]:
        return {"bc": "dirichlet", "value": strict_field_data(self.value)}

    def declaration_references(self) -> tuple[Handle, ...]:
        return collect_references(self.value)

    def resolve_references(self, resolver: Any) -> Dirichlet:
        return Dirichlet(resolve_value(self.value, resolver, where="Dirichlet value"))


class Neumann(FieldBoundaryCondition):
    def __init__(self, flux: Any = 0) -> None:
        self.flux = _boundary_value(flux)

    def options(self) -> dict[str, Any]:
        return {"bc": "neumann", "flux": strict_field_data(self.flux)}

    def declaration_references(self) -> tuple[Handle, ...]:
        return collect_references(self.flux)

    def resolve_references(self, resolver: Any) -> Neumann:
        return Neumann(resolve_value(self.flux, resolver, where="Neumann flux"))


class FirstOrderExtrapolation(FieldBoundaryCondition):
    def options(self) -> dict[str, Any]:
        return {"bc": "first_order_extrapolation"}


class BoundaryCondition(Descriptor):
    """Bind one condition to one selector; this is the sole binding authority."""

    category = "boundary_condition"

    def __init__(self, selector: Any, condition: Any) -> None:
        if not isinstance(selector, BoundarySelector):
            raise TypeError("BoundaryCondition selector must be a BoundarySelector")
        if not isinstance(condition, FieldBoundaryCondition):
            raise TypeError("BoundaryCondition condition must be a FieldBoundaryCondition")
        self.selector = selector
        self.condition = condition

    def options(self) -> dict[str, Any]:
        return {
            "selector": self.selector.options(),
            "condition": self.condition.options(),
        }

    def to_data(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "selector": self.selector.to_data(),
            "condition": self.condition.to_data(),
        }

    def declaration_references(self) -> tuple[Handle, ...]:
        return collect_references((self.selector, self.condition))

    def resolve_references(self, resolver: Any) -> BoundaryCondition:
        resolve = getattr(self.selector, "resolve_references", None)
        selector = resolve(resolver) if callable(resolve) else self.selector
        condition = resolve_value(self.condition, resolver, where="BoundaryCondition condition")
        return BoundaryCondition(selector, condition)


__all__ = [
    "AllPhysicalBoundaries",
    "AxisBoundary",
    "BoundaryCondition",
    "BoundarySelector",
    "Dirichlet",
    "FieldBoundaryCondition",
    "FirstOrderExtrapolation",
    "NamedBoundary",
    "Neumann",
    "Periodic",
    "XMax",
    "XMin",
    "YMax",
    "YMin",
    "ZMax",
    "ZMin",
]
