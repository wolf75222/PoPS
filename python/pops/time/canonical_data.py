"""Canonical immutable temporal metadata shared by authoring and graph records."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from typing import Any

from pops.identity.scalar import ScalarLiteral, scalar_data
from pops.model.handles import Handle
from pops.model.ownership import OwnerPath
from pops.time.points import Clock, StagePoint, TimePoint
from pops.time.references import handle_data


def strict_data(value: Any, *, where: str) -> Any:
    if isinstance(value, CanonicalData):
        return value.to_data()
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float, Decimal, Fraction, ScalarLiteral)):
        return {"scalar": scalar_data(value)}
    if isinstance(value, Enum):
        return {
            "enum": "%s.%s.%s"
            % (type(value).__module__, type(value).__qualname__, value.name)
        }
    if isinstance(value, OwnerPath):
        return {"owner_path": value.canonical().to_data()}
    if isinstance(value, (Clock, TimePoint, StagePoint)):
        return value.to_data()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise TypeError("%s mapping keys must be non-empty strings" % where)
        return {
            key: strict_data(item, where="%s.%s" % (where, key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            strict_data(item, where="%s[%d]" % (where, index))
            for index, item in enumerate(value)
        ]
    canonical = getattr(value, "canonical_identity", None)
    if callable(canonical):
        return strict_data(canonical(), where=where)
    to_data = getattr(value, "to_data", None)
    if callable(to_data) and getattr(value, "__pops_ir_immutable__", False) is True:
        return strict_data(to_data(), where=where)
    raise TypeError(
        "%s contains mutable/opaque %s; provide canonical immutable data"
        % (where, type(value).__name__))



@dataclass(frozen=True, slots=True, init=False)
class CanonicalData:
    """Hashable canonical-data snapshot used for semantic node metadata."""

    _json: str

    def __init__(self, value: Any, *, where: str = "ProgramGraph metadata") -> None:
        data = strict_data(value, where=where)
        object.__setattr__(
            self, "_json", json.dumps(data, sort_keys=True, separators=(",", ":")))

    def to_data(self) -> Any:
        return json.loads(self._json)



def _json_ready(value: Any) -> Any:
    if isinstance(value, Handle):
        return {"handle": handle_data(value)}
    from pops._ir.expr import Expr
    if isinstance(value, Expr):
        from pops._ir.visitors import _dag_key_data
        from pops.time.references import canonical_handle
        return _json_ready(_dag_key_data((value.resolve_references(canonical_handle),)))
    references = getattr(value, "declaration_references", None)
    resolve = getattr(value, "resolve_references", None)
    if callable(references) and references() and callable(resolve):
        from pops.time.references import canonical_handle
        value = resolve(canonical_handle)
    hook = getattr(value, "to_data", None)
    if callable(hook):
        return _json_ready(hook())
    if isinstance(value, Mapping):
        if all(isinstance(key, str) and key for key in value):
            return {key: _json_ready(item) for key, item in value.items()}
        entries = [[_json_ready(key), _json_ready(item)] for key, item in value.items()]
        entries.sort(key=lambda item: json.dumps(
            item[0], sort_keys=True, separators=(",", ":")))
        return {"mapping_entries": entries}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_json_ready(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(
            item, sort_keys=True, separators=(",", ":")))
    return value
