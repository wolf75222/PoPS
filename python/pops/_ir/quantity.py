"""Qualified scientific quantity leaves and explicit physical type metadata."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .expr import Const, Expr


@dataclass(frozen=True, slots=True)
class PhysicalDimension:
    """Known physical exponents; the empty tuple means explicitly dimensionless.

    ``None`` on a quantity means unknown, never dimensionless. These are dimensions,
    not a unit conversion system: numerical values retain the author's chosen units.
    """
    powers: tuple[tuple[str, Fraction], ...] = ()
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        items = tuple((name, Fraction(power)) for name, power in self.powers)
        if any(not isinstance(name, str) or not name for name, _ in items):
            raise TypeError("physical dimensions require named base dimensions")
        if len({name for name, _ in items}) != len(items):
            raise ValueError("physical dimension bases must be unique")
        object.__setattr__(self, "powers", tuple(sorted((n, p) for n, p in items if p)))

    def to_data(self) -> dict[str, Any]:
        return {"kind": "physical_dimension", "powers": [
            [name, power.numerator, power.denominator] for name, power in self.powers]}

    @classmethod
    def from_data(cls, data: Any) -> PhysicalDimension:
        if not isinstance(data, dict) or set(data) != {"kind", "powers"} \
                or data["kind"] != "physical_dimension":
            raise ValueError("invalid physical dimension data")
        return cls(tuple((name, Fraction(n, d)) for name, n, d in data["powers"]))


@dataclass(frozen=True, slots=True)
class PhysicalSupport:
    """Mathematical coordinate domains, independent of array shape and layout."""
    coordinates: tuple[tuple[str, str], ...]
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        values = tuple(tuple(item) for item in self.coordinates)
        if not values or any(len(item) != 2 or any(
                not isinstance(value, str) or not value for value in item) for item in values):
            raise TypeError("support coordinates require (name, domain) pairs")
        if len({name for name, _ in values}) != len(values):
            raise ValueError("support coordinate names must be unique")
        object.__setattr__(self, "coordinates", values)

    def to_data(self) -> dict[str, Any]:
        return {"kind": "physical_support", "coordinates": [list(x) for x in self.coordinates]}

    @classmethod
    def from_data(cls, data: Any) -> PhysicalSupport:
        if not isinstance(data, dict) or set(data) != {"kind", "coordinates"} \
                or data["kind"] != "physical_support":
            raise ValueError("invalid physical support data")
        return cls(tuple(tuple(item) for item in data["coordinates"]))


_hash_owner: ContextVar[Any] = ContextVar("pops_expression_hash_owner", default=None)


@contextmanager
def local_expression_identity(owner: Any):
    """Normalize only a defining Module's own leaves while computing its content hash."""
    token = _hash_owner.set(owner)
    try:
        yield
    finally:
        _hash_owner.reset(token)


def expression_handle_key(handle: Any) -> tuple[Any, ...]:
    if handle.owner_path == _hash_owner.get():
        return ("local", handle.kind, handle.local_id,
                getattr(handle, "registered_operator_name", None))
    return ("qualified", handle.qualified_id)


class QuantityRef(Expr):
    """One component of an authenticated declaration, kept typed until target binding."""

    def __init__(self, handle: Any, component: str, *, space: Any = None) -> None:
        from pops.model.handles import Handle
        from pops.model.spaces import Space
        if not isinstance(handle, Handle) or handle.kind not in ("state", "field"):
            raise TypeError("QuantityRef requires a state or field declaration Handle")
        space = getattr(handle, "space", None) if space is None else space
        if not isinstance(space, Space) or component not in space.components:
            raise ValueError("QuantityRef component must belong to its typed Space")
        if handle.local_id != space.name and not handle.is_instance:
            raise ValueError("QuantityRef handle and Space declaration disagree")
        declared_space = getattr(handle, "space", space)
        if declared_space != space:
            raise ValueError("QuantityRef cannot change the declaration's physical type")
        self.handle = handle
        self.space = space
        self.component = component
        self.index = space.components.index(component)

    @property
    def qualified_id(self) -> str:
        return "%s::component::%d" % (self.handle.qualified_id, self.index)

    def eval(self, env: Any) -> Any:
        if self.qualified_id in env:
            return env[self.qualified_id]
        if self.handle.qualified_id in env:
            return env[self.handle.qualified_id][self.index]
        raise KeyError("qualified quantity %s has no value" % self.qualified_id)

    def deps(self) -> set[str]:
        return {self.qualified_id}

    def __pops_ir_children__(self) -> tuple:
        return ()

    def __pops_ir_key__(self, recurse: Any) -> Any:
        return ("quantity", expression_handle_key(self.handle), self.index, self.space._key())

    def __pops_ir_diff__(self, *, recurse: Any, target: Any, definitions: Any) -> Any:
        if isinstance(target, QuantityRef):
            return Const(int(self.handle == target.handle and self.index == target.index))
        return Const(0)

    def to_cpp(self) -> str:
        raise TypeError("QuantityRef requires an authenticated owner-aware native binding")

    def to_data(self) -> dict[str, Any]:
        handle = ({"kind": self.handle.kind, "local_id": self.handle.local_id}
                  if self.handle.owner_path == _hash_owner.get()
                  else self.handle.canonical_identity())
        return {"kind": "quantity_ref", "handle": handle,
                "component": self.component, "space": self.space.to_data()}

    def _str(self) -> str:
        return "quantity(%s, %s)" % (self.handle.local_id, self.component)


__all__ = ["PhysicalDimension", "PhysicalSupport", "QuantityRef"]
