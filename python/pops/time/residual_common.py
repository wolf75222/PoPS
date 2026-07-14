"""Shared canonical-data primitives for residual descriptors."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import fields
from typing import Any, ClassVar

from pops._ir.literals import ScalarLiteral, scalar_data


def residual_name(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("%s must be a non-empty, trimmed string" % where)
    return value


def residual_names(
        values: Iterable[Any], where: str, *, nonempty: bool = False) -> tuple[str, ...]:
    result = tuple(residual_name(value, where) for value in values)
    if nonempty and not result:
        raise ValueError("%s must not be empty" % where)
    if len(set(result)) != len(result):
        raise ValueError("%s contains duplicate names" % where)
    return result


def coverage_errors(available: set[str], covered: set[str], label: str) -> tuple[str, ...]:
    missing = available - covered
    if not missing:
        return ()
    return ("terms must cover every %s component: %s" % (
        label, ", ".join(sorted(missing))),)


def _data(value: Any) -> Any:
    if isinstance(value, CanonicalDescriptor):
        return value.to_data()
    if isinstance(value, Mapping):
        return {key: _data(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_data(item) for item in value]
    if isinstance(value, ScalarLiteral):
        return scalar_data(value)
    return value


class CanonicalDescriptor:
    """Mixin for immutable JSON-ready descriptor data."""

    __pops_ir_immutable__: ClassVar[bool] = True
    kind: ClassVar[str]

    def to_data(self) -> dict[str, Any]:
        return {"kind": self.kind, **{
            field.name: _data(getattr(self, field.name)) for field in fields(self)}}


__all__ = ["CanonicalDescriptor", "coverage_errors", "residual_name", "residual_names"]
