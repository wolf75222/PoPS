"""Small immutable elementwise vector algebra over the common scalar Expr nodes."""
from __future__ import annotations

from typing import Any

from .expr import Expr, _wrap


class VectorExpr(tuple):
    """An ordered, fixed-width expression vector with scalar broadcasting.

    Its elements remain ordinary Expr leaves, so codegen and dependency discovery
    use the existing scalar DAG. Indexing and iteration retain the component order.
    """

    def __new__(cls, values: Any) -> VectorExpr:
        items = tuple(_wrap(value) for value in values)
        if not items or any(not isinstance(item, Expr) for item in items):
            raise TypeError("VectorExpr needs one or more scalar Expr components")
        return tuple.__new__(cls, items)

    def __bool__(self) -> bool:
        raise TypeError("symbolic VectorExpr has no Python truth value")

    def _binary(self, other: Any, operation: str) -> VectorExpr:
        if isinstance(other, (tuple, list)):
            if len(other) != len(self):
                raise ValueError("VectorExpr operands have different component counts")
            right = tuple(_wrap(value) for value in other)
        else:
            right = (_wrap(other),) * len(self)
        return VectorExpr(getattr(a, operation)(b) for a, b in zip(self, right, strict=True))

    def __add__(self, other: Any) -> VectorExpr:
        return self._binary(other, "__add__")

    def __radd__(self, other: Any) -> VectorExpr:
        return self._binary(other, "__radd__")

    def __sub__(self, other: Any) -> VectorExpr:
        return self._binary(other, "__sub__")

    def __rsub__(self, other: Any) -> VectorExpr:
        return self._binary(other, "__rsub__")

    def __mul__(self, other: Any) -> VectorExpr:
        return self._binary(other, "__mul__")

    def __rmul__(self, other: Any) -> VectorExpr:
        return self._binary(other, "__rmul__")

    def __truediv__(self, other: Any) -> VectorExpr:
        return self._binary(other, "__truediv__")

    def __rtruediv__(self, other: Any) -> VectorExpr:
        return self._binary(other, "__rtruediv__")

    def __neg__(self) -> VectorExpr:
        return VectorExpr(-item for item in self)


__all__ = ["VectorExpr"]
