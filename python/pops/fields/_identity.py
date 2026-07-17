"""Strict structural data for public field contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from pops.descriptors import Descriptor
from pops.identity import Identity, make_identity
from pops._ir.elliptic import CoeffGradient, DivCoeffGrad, EllipticSum, Reaction
from pops._ir.expr import Expr, Gradient, Laplacian, Partial
from pops.math import Equation
from pops.model import Handle
from pops.model.hash_data import canonical_hash_data


_FIELD_EXPR_PROJECTIONS: Mapping[type[Any], Callable[[Any], Any]] = MappingProxyType(
    {
        Laplacian: lambda value: {"field": value.field, "scale": value.scale},
        Gradient: lambda value: {"field": value.field, "scale": value.scale},
        Partial: lambda value: {
            "field": value.field,
            "axis": value.axis,
            "scale": value.scale,
        },
        CoeffGradient: lambda value: {
            "field": value.field,
            "coefficient": value.coeff,
            "scale": value.scale,
        },
        DivCoeffGrad: lambda value: {
            "field": value.field,
            "coefficient": value.coeff,
            "scale": value.scale,
        },
        Reaction: lambda value: {
            "field": value.field,
            "coefficient": value.coeff,
            "scale": value.scale,
        },
        EllipticSum: lambda value: {"terms": value.terms},
    }
)


def strict_field_data(value: Any) -> Any:
    """Project supported field values without ``repr`` or address identity."""
    if isinstance(value, Handle):
        return {"handle": value.canonical_identity()}
    if isinstance(value, Identity):
        return {"identity": value.token}
    if isinstance(value, Equation):
        return {
            "equation": {
                "lhs": strict_field_data(value.lhs),
                "rhs": strict_field_data(value.rhs),
            }
        }
    if isinstance(value, Expr):
        projector = _FIELD_EXPR_PROJECTIONS.get(type(value))
        if projector is not None:
            return {
                "field_expression": {
                    "type": "%s.%s" % (type(value).__module__, type(value).__qualname__),
                    "data": strict_field_data(projector(value)),
                }
            }
        return strict_field_data(canonical_hash_data(value, where="field expression"))
    hook = getattr(value, "to_data", None)
    if callable(hook):
        return strict_field_data(hook())
    if isinstance(value, Descriptor) or hasattr(value, "category"):
        raise TypeError(
            "field descriptor %s must implement the small exact to_data() protocol"
            % type(value).__name__
        )
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise TypeError("field identity mappings require non-empty string keys")
        return {key: strict_field_data(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [strict_field_data(item) for item in value]
    if isinstance(value, (set, frozenset)):
        projected = [strict_field_data(item) for item in value]
        return sorted(projected, key=lambda item: make_identity("field-set-item", item).token)
    if isinstance(value, float):
        return {"binary64": value.hex()}
    if isinstance(value, bytes):
        raise TypeError("field identity is strict JSON and refuses bytes")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError(
        "field identity contains opaque %s; use a typed descriptor, Handle, Expr, or to_data()"
        % type(value).__name__
    )


def field_identity(domain: str, payload: Any) -> Identity:
    return make_identity(domain, strict_field_data(payload))


__all__ = [
    "field_identity",
    "strict_field_data",
]
