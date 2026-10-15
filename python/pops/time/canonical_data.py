"""Canonical immutable temporal metadata shared by authoring and graph records."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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
    from pops._structural_data import transform_tree
    def describe(node):
        if isinstance(node, CanonicalData):
            return None, node.to_data()
        if node is None or isinstance(node, (bool, str)):
            return None, node
        if isinstance(node, (int, float, Decimal, Fraction, ScalarLiteral)):
            return None, {"scalar": scalar_data(node)}
        if isinstance(node, Enum):
            return None, {"enum": "%s.%s.%s" % (
                type(node).__module__, type(node).__qualname__, node.name)}
        if isinstance(node, OwnerPath):
            return None, {"owner_path": node.canonical().to_data()}
        if isinstance(node, (Clock, TimePoint, StagePoint)):
            return None, node.to_data()
        if isinstance(node, Mapping):
            if any(not isinstance(key, str) or not key for key in node):
                raise TypeError("%s mapping keys must be non-empty strings" % where)
            keys = tuple(node)
            return tuple(node.values()), lambda children: dict(zip(keys, children, strict=False))
        if isinstance(node, (list, tuple)):
            return node, list
        canonical = getattr(node, "canonical_identity", None)
        if callable(canonical):
            return (canonical(),), lambda children: children[0]
        to_data = getattr(node, "to_data", None)
        if callable(to_data) and getattr(node, "__pops_ir_immutable__", False) is True:
            return (to_data(),), lambda children: children[0]
        raise TypeError("%s contains mutable/opaque %s; provide canonical immutable data"
                        % (where, type(node).__name__))
    return transform_tree(value, describe)



@dataclass(frozen=True, slots=True, init=False)
class CanonicalData:
    """Hashable canonical-data snapshot used for semantic node metadata."""

    _json: str
    _data: Any = field(compare=False, hash=False, repr=False)

    def __init__(self, value: Any, *, where: str = "ProgramGraph metadata") -> None:
        data = strict_data(value, where=where)
        from types import MappingProxyType
        from pops._structural_data import canonical_json, transform
        def describe(node):
            if isinstance(node, Mapping):
                keys = tuple(node)
                return tuple(node.values()), lambda children: MappingProxyType(dict(zip(keys, children, strict=False)))
            if isinstance(node, (list, tuple)):
                return node, tuple
            return None, node
        object.__setattr__(self, "_json", canonical_json(data))
        object.__setattr__(self, "_data", transform(data, describe))

    def to_data(self) -> Any:
        return _json_ready(self._data)



def _json_ready(value: Any) -> Any:
    from pops._structural_data import canonical_json, transform_tree, _plain_json_data
    if type(value) in (dict, list, tuple) and _plain_json_data(
            value, nonempty_keys=True, finite_floats=True):
        import json
        try:
            # C codecs copy each occurrence independently, just like the tree
            # traversal below. No aliases, hooks, keys or scalar types change.
            return json.loads(json.dumps(value, separators=(",", ":")))
        except (RecursionError, ValueError):
            # Deep trees and integers beyond the JSON digit optimization use
            # the same unrestricted projection and diagnostics as before.
            pass
    def describe(node):
        value_type = type(node)
        if node is None or (value_type is bool or value_type is int or value_type is float or value_type is str):
            return None, node
        if value_type is not dict and value_type is not list and value_type is not tuple:
            if isinstance(node, Handle):
                return None, {"handle": handle_data(node)}
            from pops._ir.expr import Expr
            if isinstance(node, Expr):
                from pops._ir.visitors import _dag_key_data
                from pops.time.references import canonical_handle
                data = _dag_key_data((node.resolve_references(canonical_handle),))
                return (data,), lambda children: children[0]
            references = getattr(node, "declaration_references", None)
            resolve = getattr(node, "resolve_references", None)
            if callable(references) and references() and callable(resolve):
                from pops.time.references import canonical_handle
                node = resolve(canonical_handle)
            hook = getattr(node, "to_data", None)
            if callable(hook):
                return (hook(),), lambda children: children[0]
        if isinstance(node, Mapping):
            keys = tuple(node)
            if all(isinstance(key, str) and key for key in keys):
                return tuple(node.values()), lambda children: dict(zip(keys, children, strict=False))
            flat = tuple(item for pair in node.items() for item in pair)
            def entries(children):
                rows = [children[i:i + 2] for i in range(0, len(children), 2)]
                rows.sort(key=lambda row: canonical_json(row[0]))
                return {"mapping_entries": rows}
            return flat, entries
        if isinstance(node, (list, tuple)):
            return node, list
        if isinstance(node, (set, frozenset)):
            return tuple(node), lambda children: sorted(children, key=canonical_json)
        return None, node
    return transform_tree(value, describe)
