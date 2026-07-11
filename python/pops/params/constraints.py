"""Strict, serializable parameter-domain constraints."""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from fractions import Fraction
from typing import Any

from pops.descriptors import Descriptor


def _literal_data(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"kind": "string", "value": value}
    if isinstance(value, bool):
        return {"kind": "boolean", "value": value}
    if value is None:
        return {"kind": "none"}
    from pops.ir.literals import scalar_literal

    return scalar_literal(value).to_data()


def _literal_from_data(data: Any) -> Any:
    if not isinstance(data, Mapping) or not isinstance(data.get("kind"), str):
        raise TypeError("constraint literal data must be a mapping with a kind")
    kind = data["kind"]
    if kind == "string" and set(data) == {"kind", "value"} and isinstance(data["value"], str):
        return data["value"]
    if kind == "boolean" and set(data) == {"kind", "value"} and isinstance(data["value"], bool):
        return data["value"]
    if kind == "none" and set(data) == {"kind"}:
        return None
    if kind == "integer" and set(data) in ({"kind", "value"}, {"kind", "value", "target"}):
        return int(data["value"])
    if kind == "rational" and set(data) in (
        {"kind", "numerator", "denominator"},
        {"kind", "numerator", "denominator", "target"},
    ):
        return Fraction(int(data["numerator"]), int(data["denominator"]))
    if kind == "decimal" and set(data) in ({"kind", "value"}, {"kind", "value", "target"}):
        return Decimal(data["value"])
    if kind == "binary64" and set(data) in ({"kind", "value"}, {"kind", "value", "target"}):
        return float.fromhex(data["value"])
    raise TypeError("unsupported canonical constraint literal %r" % dict(data))


class Constraint(Descriptor):
    category = "constraint"
    constraint_kind = "any"

    def check(self, value: Any, who: str = "value") -> bool:
        return True

    def to_data(self) -> dict[str, Any]:
        return {"kind": self.constraint_kind}

    @classmethod
    def from_data(cls, data: Any) -> Constraint:
        if not isinstance(data, Mapping) or not isinstance(data.get("kind"), str):
            raise TypeError("Constraint data must be a mapping with a kind")
        kind = data["kind"]
        if kind == "positive" and set(data) == {"kind"}:
            return Positive()
        if kind == "non_negative" and set(data) == {"kind"}:
            return NonNegative()
        if kind in ("range", "interval") and set(data) == {"kind", "lo", "hi"}:
            target = Interval if kind == "interval" else Range
            return target(_literal_from_data(data["lo"]), _literal_from_data(data["hi"]))
        if kind in ("in", "one_of") and set(data) == {"kind", "allowed"}:
            if not isinstance(data["allowed"], (tuple, list)):
                raise TypeError("Constraint allowed must be a list")
            target = OneOf if kind == "one_of" else In
            return target(*(_literal_from_data(item) for item in data["allowed"]))
        raise TypeError("unsupported canonical Constraint data %r" % dict(data))

    def options(self) -> dict[str, Any]:
        data = self.to_data()
        return {key: value for key, value in data.items() if key != "kind"}

    def freeze(self) -> Constraint:
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("%s is immutable" % type(self).__name__)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("%s is immutable" % type(self).__name__)

    def __eq__(self, other: Any) -> bool:
        return type(self) is type(other) and self.to_data() == other.to_data()

    def __hash__(self) -> int:
        return hash(repr(self.to_data()))


class Positive(Constraint):
    constraint_kind = "positive"

    def check(self, value: Any, who: str = "value") -> bool:
        if value <= 0:
            raise ValueError("%s must be > 0 (got %r)" % (who, value))
        return True


class NonNegative(Constraint):
    constraint_kind = "non_negative"

    def check(self, value: Any, who: str = "value") -> bool:
        if value < 0:
            raise ValueError("%s must be >= 0 (got %r)" % (who, value))
        return True


class Range(Constraint):
    constraint_kind = "range"

    def __init__(self, lo: Any, hi: Any) -> None:
        _literal_data(lo)
        _literal_data(hi)
        if lo > hi:
            raise ValueError("Range: lo must be <= hi (got lo=%r hi=%r)" % (lo, hi))
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    def to_data(self) -> dict[str, Any]:
        return {"kind": self.constraint_kind, "lo": _literal_data(self.lo),
                "hi": _literal_data(self.hi)}

    def check(self, value: Any, who: str = "value") -> bool:
        if not self.lo <= value <= self.hi:
            raise ValueError("%s must be in [%r, %r] (got %r)" % (who, self.lo, self.hi, value))
        return True


class In(Constraint):
    """Membership in a fixed, serializable set of allowed values."""

    constraint_kind = "in"

    def __init__(self, *allowed: Any) -> None:
        if not allowed:
            raise ValueError("In requires at least one allowed value")
        for value in allowed:
            _literal_data(value)
        object.__setattr__(self, "allowed", tuple(allowed))

    def to_data(self) -> dict[str, Any]:
        return {"kind": self.constraint_kind,
                "allowed": [_literal_data(value) for value in self.allowed]}

    def check(self, value: Any, who: str = "value") -> bool:
        if value not in self.allowed:
            raise ValueError("%s must be one of %r (got %r)" % (who, self.allowed, value))
        return True


class Interval(Range):
    constraint_kind = "interval"


class OneOf(In):
    constraint_kind = "one_of"


__all__ = ["Constraint", "Positive", "NonNegative", "Range", "In", "Interval", "OneOf"]
