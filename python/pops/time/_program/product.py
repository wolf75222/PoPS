"""``Product(name=value, ...)``: the record of named values of the temporal program (IR1).

A leaf of the temporal package: the Outputs (CC14), the method memory (T27) and the traced
methods (``Candidate(memory=...)``) all build records, and none of them is imported here.
"""
from __future__ import annotations

import keyword
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any


class Product(Mapping):
    """``Product(name=value, ...)``: a record of named values, in the order written (IR1)."""

    __slots__ = ("_fields",)

    def __init__(self, **fields: Any) -> None:
        for name in fields:
            if not name.isidentifier() or keyword.iskeyword(name):
                raise ValueError("Product field names are identifiers, got %r" % (name,))
        object.__setattr__(self, "_fields", MappingProxyType(dict(fields)))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Product is immutable")

    def __getitem__(self, name: str) -> Any:
        return self._fields[name]

    def __getattr__(self, name: str) -> Any:
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(name) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __repr__(self) -> str:
        return "Product(%s)" % ", ".join("%s=%r" % item for item in self._fields.items())


__all__ = ["Product"]
