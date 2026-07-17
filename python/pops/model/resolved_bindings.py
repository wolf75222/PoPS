"""Immutable effective values produced by a BindSchema resolution."""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any, cast

from .handles import ParamHandle


_SOURCES = frozenset({"override", "default", "const", "derived"})


class ResolvedBindings(Mapping[ParamHandle, Any]):
    """Dict-compatible values plus their exact schema and materialization source."""

    __slots__ = ("schema", "values", "sources")

    def __init__(self, schema: Any, values: Any, sources: Any) -> None:
        if not isinstance(values, Mapping) or not isinstance(sources, Mapping):
            raise TypeError("ResolvedBindings values and sources must be mappings")
        if set(values) != set(sources):
            raise ValueError("ResolvedBindings requires exactly one source for every value")
        checked_values = {}
        checked_sources = {}
        for handle, value in values.items():
            if not isinstance(handle, ParamHandle) or not handle.is_resolved:
                raise TypeError("ResolvedBindings keys must be canonical ParamHandle values")
            source = sources[handle]
            if source not in _SOURCES:
                raise ValueError(
                    "ResolvedBindings source must be one of %s (got %r)"
                    % (sorted(_SOURCES), source)
                )
            checked_values[handle] = value
            checked_sources[handle] = source
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "values", MappingProxyType(checked_values))
        object.__setattr__(self, "sources", MappingProxyType(checked_sources))

    def _value_map(self) -> Mapping[ParamHandle, Any]:
        """Typed view of the deliberately published immutable ``values`` table."""
        return cast(Mapping[ParamHandle, Any], object.__getattribute__(self, "values"))

    def _source_map(self) -> Mapping[ParamHandle, str]:
        """Typed view of the immutable materialization-source table."""
        return cast(Mapping[ParamHandle, str], object.__getattribute__(self, "sources"))

    def __getitem__(self, handle: ParamHandle) -> Any:
        return self._value_map()[handle]

    def __iter__(self) -> Iterator[ParamHandle]:
        return iter(self._value_map())

    def __len__(self) -> int:
        return len(self._value_map())

    def source(self, handle: ParamHandle) -> str:
        return self._source_map()[handle]

    def rows(self) -> tuple[dict[str, Any], ...]:
        """JSON-ready effective-value rows in stable schema slot order."""
        from pops.params._declaration_data import value_data
        from ._bind_schema_data import dtype_object

        rows = []
        for slot in self.schema.slots:
            declaration = slot.declaration
            rows.append({
                "ordinal": slot.ordinal,
                "qid": slot.qid,
                "handle": slot.handle.canonical_identity(),
                "kind": slot.kind,
                "dtype": slot.dtype,
                "unit": declaration["unit"],
                "source": self._source_map()[slot.handle],
                "value": value_data(
                    self._value_map()[slot.handle],
                    dtype=dtype_object(slot.dtype),
                    unit=declaration["unit"],
                    where="resolved binding %s" % slot.qid,
                ),
                "provenance": declaration["provenance"],
            })
        return tuple(rows)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("ResolvedBindings is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("ResolvedBindings is immutable")

    def __repr__(self) -> str:
        return "ResolvedBindings(values=%d, schema_hash=%r)" % (
            len(self), self.schema.hash[:12]
        )


__all__ = ["ResolvedBindings"]
