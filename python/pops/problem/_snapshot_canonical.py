"""Strict canonical projections used by :mod:`pops.problem._snapshot`.

This module contains the array-free, authenticated conversion of authoring values into stable
JSON-ready data.  It is deliberately independent from snapshot lifecycle and hashing.
"""
from __future__ import annotations

import json
import math
import types
from collections.abc import Mapping
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from typing import Any
from urllib.parse import quote

from pops._ir.literals import ScalarLiteral
from pops.model.handles import Handle, OperatorHandle as ModelOperatorHandle, OwnerPath, ParamHandle
from pops.problem._snapshot_callable import (
    canonical_callable_reference as _canonical_callable_reference,
    class_has_public_behavior as _class_has_public_behavior,
    class_implementation_projection as _class_implementation_projection,
)
from pops.problem._snapshot_mapping import canonical_mapping as _canonical_mapping
from pops.problem._snapshot_literals import canonical_enum_data as _canonical_enum_data, canonical_literal_data as _canonical_literal_data


def _canonical(
    value: Any,
    *,
    path: str = "$",
    active: set[int] | None = None,
    handle_resolver: Any = None,
    artifact: bool = False,
    projection_cache: dict[tuple[int, str], Any] | None = None,
) -> Any:
    """Return a strict, deterministic JSON view of ``value``.

    No lossy fallback exists.  An object is encoded from structural projections and its qualified
    Python type, or the snapshot fails at the exact path.  ``active`` detects reference cycles while
    still allowing the same immutable descriptor to appear in two independent branches.
    """
    if active is None:
        active = set()
    if projection_cache is None:
        projection_cache = {}
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return {"$scalar": {"kind": "integer", "value": str(value)}}
    if isinstance(value, Fraction):
        return {"$scalar": {"kind": "rational", "numerator": str(value.numerator),
                            "denominator": str(value.denominator)}}
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("AuthoringSnapshot cannot encode a non-finite Decimal")
        return {"$scalar": {"kind": "decimal", "value": str(value)}}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("AuthoringSnapshot cannot encode a non-finite float at %s" % path)
        return {"$scalar": {"kind": "binary64", "value": value.hex()}}
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return {"$bytes": value.hex()}
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("AuthoringSnapshot cannot encode a non-finite complex at %s" % path)
        return {"$complex": [value.real.hex(), value.imag.hex()]}
    if isinstance(value, Enum):
        return _canonical_enum_data(value, path=path)
    if isinstance(value, types.ModuleType):
        raise TypeError(
            "AuthoringSnapshot cannot encode module %s as an opaque value at %s; "
            "reference explicit module attributes from a callable instead"
            % (getattr(value, "__name__", "<module>"), path))
    if isinstance(value, type):
        projection = _class_implementation_projection(
            value,
            path=path,
            active=active,
            handle_resolver=handle_resolver,
            artifact=artifact,
            canonical=lambda item, **kwargs: _canonical(
                item, projection_cache=projection_cache, **kwargs),
            dependency_cache=projection_cache,
        )
        return {"$class": projection}
    if isinstance(value, (types.FunctionType, types.MethodType, types.BuiltinFunctionType)):
        return _canonical_callable_reference(
            value,
            path=path,
            active=active,
            handle_resolver=handle_resolver,
            artifact=artifact,
            canonical=lambda item, **kwargs: _canonical(
                item, projection_cache=projection_cache, **kwargs),
            dependency_cache=projection_cache,
        )
    marker = id(value)
    if marker in active:
        raise ValueError("AuthoringSnapshot cannot encode a reference cycle at %s" % path)
    active.add(marker)
    try:
        return _canonical_compound(
            value, path=path, active=active, handle_resolver=handle_resolver,
            artifact=artifact, projection_cache=projection_cache)
    finally:
        active.remove(marker)


def _canonical_compound(
    value: Any,
    *,
    path: str,
    active: set[int],
    handle_resolver: Any,
    artifact: bool,
    projection_cache: dict[tuple[int, str], Any],
) -> Any:
    """Canonicalise a container, handle, literal or structural Python object."""
    if type(value).__module__.split(".", 1)[0] == "numpy":
        raise TypeError(
            "AuthoringSnapshot cannot encode numpy runtime data at %s; declare metadata, not arrays"
            % path)
    if isinstance(value, list):
        return [_canonical(
                    item, path="%s[%d]" % (path, index), active=active,
                    handle_resolver=handle_resolver, artifact=artifact,
                    projection_cache=projection_cache)
                for index, item in enumerate(value)]
    if isinstance(value, tuple):
        return {"$tuple": [_canonical(
                    item, path="%s[%d]" % (path, index), active=active,
                    handle_resolver=handle_resolver, artifact=artifact,
                    projection_cache=projection_cache)
                for index, item in enumerate(value)]}
    if isinstance(value, set):
        items = [_canonical(
                    item, path="%s{item}" % path, active=active,
                    handle_resolver=handle_resolver, artifact=artifact,
                    projection_cache=projection_cache)
                 for item in value]
        return {"$set": sorted(items, key=lambda item: json.dumps(
            item, sort_keys=True, separators=(",", ":"), allow_nan=False))}
    if isinstance(value, frozenset):
        items = [_canonical(
                    item, path="%s{item}" % path, active=active,
                    handle_resolver=handle_resolver, artifact=artifact,
                    projection_cache=projection_cache)
                 for item in value]
        return {"$frozenset": sorted(items, key=lambda item: json.dumps(
            item, sort_keys=True, separators=(",", ":"), allow_nan=False))}
    if isinstance(value, Mapping):
        return _canonical_mapping(
            value,
            path=path,
            active=active,
            handle_resolver=handle_resolver,
            artifact=artifact,
            canonical=lambda item, **kwargs: _canonical(
                item, projection_cache=projection_cache, **kwargs),
        )
    if isinstance(value, OwnerPath):
        canonical_owner = value.canonical()
        data = _call_projection(
            canonical_owner, "to_data", canonical_owner.to_data, path, projection_cache)
        if OwnerPath.from_data(data) != canonical_owner:
            raise ValueError("OwnerPath.to_data() does not round-trip at %s" % path)
        return {"$owner_path": data}
    if isinstance(value, Handle):
        return {"$handle": _canonical_handle_identity(
            value, path, handle_resolver=handle_resolver,
            projection_cache=projection_cache)}
    if isinstance(value, ScalarLiteral):
        data = _call_projection(value, "to_data", value.to_data, path, projection_cache)
        return {"$scalar": _canonical_literal_data(data, path="%s.to_data()" % path)}
    return _canonical_object(
        value, path=path, active=active, handle_resolver=handle_resolver,
        artifact=artifact, projection_cache=projection_cache)


def _canonical_handle_identity(
    value: Handle,
    path: str,
    *,
    handle_resolver: Any = None,
    projection_cache: dict[tuple[int, str], Any],
) -> dict[str, Any]:
    """Validate an authenticated Handle projection without lossy coercion or duck typing."""
    if callable(handle_resolver):
        authored = value
        resolved = handle_resolver(value)
        if not isinstance(resolved, Handle) or not resolved.is_resolved:
            raise TypeError(
                "AuthoringSnapshot handle resolver must return a canonical Handle at %s" % path)
        if (resolved.schema_version, resolved.kind, resolved.local_id) != (
                authored.schema_version, authored.kind, authored.local_id):
            raise ValueError(
                "AuthoringSnapshot handle resolver changed declaration identity at %s" % path)
        if isinstance(authored, ModelOperatorHandle) and (
                not isinstance(resolved, ModelOperatorHandle)
                or resolved.registered_operator_name != authored.registered_operator_name):
            raise ValueError(
                "AuthoringSnapshot handle resolver changed operator target at %s" % path)
        if isinstance(authored, ParamHandle) and (
                not isinstance(resolved, ParamHandle)
                or resolved.param_kind != authored.param_kind):
            raise ValueError(
                "AuthoringSnapshot handle resolver changed parameter kind at %s" % path)
        if authored.is_resolved \
                and resolved.canonical_identity() != authored.canonical_identity():
            raise ValueError(
                "AuthoringSnapshot handle resolver changed an already canonical identity at %s"
                % path)
        value = resolved
    elif not value.is_resolved:
        raise TypeError(
            "AuthoringSnapshot cannot serialize unresolved handle %s at %s without an "
            "authoritative resolver" % (value.qualified_id, path))
    identity = _call_projection(
        value, "canonical_identity", value.canonical_identity, path, projection_cache)
    if not isinstance(identity, Mapping):
        raise TypeError("Handle.canonical_identity() must return a mapping at %s" % path)
    required = {"qualified_id", "schema_version", "kind", "owner_path", "local_id"}
    allowed = set(required)
    if isinstance(value, ModelOperatorHandle):
        allowed.add("registered_operator_name")
    if isinstance(value, ParamHandle):
        allowed.update(("handle_type", "param_kind"))
    from pops.problem.handles import BlockHandle
    if isinstance(value, BlockHandle):
        allowed.update(("handle_type", "model_owner_path"))
    if value.is_instance:
        allowed.update(("declaration_ref", "block_ref"))
    if set(identity) != allowed:
        raise TypeError(
            "Handle.canonical_identity() keys at %s must be exactly %s (got %s)"
            % (path, sorted(allowed), sorted(identity)))
    schema_version = identity["schema_version"]
    kind = identity["kind"]
    local_id = identity["local_id"]
    qualified_id = identity["qualified_id"]
    owner_data = identity["owner_path"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) \
            or schema_version < 1:
        raise TypeError("Handle canonical schema_version must be an integer >= 1 at %s" % path)
    for name, item in (("kind", kind), ("local_id", local_id),
                       ("qualified_id", qualified_id)):
        if not isinstance(item, str) or not item:
            raise TypeError("Handle canonical %s must be a non-empty string at %s" % (name, path))
    if not isinstance(value.owner_path, OwnerPath):
        raise TypeError("Handle owner_path must be an authenticated OwnerPath at %s" % path)
    try:
        owner = OwnerPath.from_data(owner_data)
    except (TypeError, ValueError) as exc:
        raise TypeError("Handle canonical owner_path is invalid at %s" % path) from exc
    if (schema_version, kind, local_id, owner) != (
            value.schema_version, value.kind, value.local_id, value.owner_path):
        raise ValueError("Handle.canonical_identity() disagrees with its immutable fields at %s"
                         % path)

    expected_qualified = Handle._qualified_id(value, owner)
    if isinstance(value, ModelOperatorHandle):
        target = identity["registered_operator_name"]
        if not isinstance(target, str) or not target:
            raise TypeError(
                "Handle canonical registered_operator_name must be a non-empty string at %s" % path)
        if target != value.registered_operator_name:
            raise ValueError(
                "Handle.canonical_identity() disagrees with its registered target at %s" % path)
        expected_qualified = "%s::target::%s" % (expected_qualified, quote(target, safe=""))
    if isinstance(value, ParamHandle):
        if identity["handle_type"] != "parameter":
            raise ValueError("ParamHandle canonical handle_type must be 'parameter' at %s" % path)
        param_kind = identity["param_kind"]
        if param_kind != value.param_kind:
            raise ValueError(
                "ParamHandle canonical param_kind disagrees with its immutable field at %s" % path)
        expected_qualified = value._qualified_param_id(owner)
    if qualified_id != expected_qualified:
        raise ValueError("Handle.canonical_identity() has an invalid qualified_id at %s" % path)
    result = {
        "qualified_id": qualified_id,
        "schema_version": schema_version,
        "kind": kind,
        "owner_path": owner_data,
        "local_id": local_id,
    }
    if isinstance(value, ModelOperatorHandle):
        result["registered_operator_name"] = identity["registered_operator_name"]
    if isinstance(value, ParamHandle):
        result["handle_type"] = "parameter"
        result["param_kind"] = identity["param_kind"]
    if isinstance(value, BlockHandle):
        if identity["handle_type"] != "block":
            raise ValueError("BlockHandle canonical handle_type must be 'block' at %s" % path)
        try:
            model_owner = OwnerPath.from_data(identity["model_owner_path"])
        except (TypeError, ValueError) as exc:
            raise TypeError("BlockHandle model_owner_path is invalid at %s" % path) from exc
        if model_owner != value.model_owner_path:
            raise ValueError(
                "BlockHandle canonical model_owner_path disagrees with its immutable field at %s"
                % path)
        result["handle_type"] = "block"
        result["model_owner_path"] = identity["model_owner_path"]
    if value.is_instance:
        for ref_name in ("declaration_ref", "block_ref"):
            ref_identity = identity[ref_name]
            decoded = Handle.from_canonical_identity(ref_identity)
            if decoded.canonical_identity() != ref_identity:
                raise ValueError(
                    "Handle canonical %s does not round-trip at %s" % (ref_name, path))
            result[ref_name] = ref_identity
    if Handle.from_canonical_identity(result).canonical_identity() != result:
        raise ValueError("Handle canonical identity does not round-trip at %s" % path)
    return result


def _canonical_object(
    value: Any,
    *,
    path: str,
    active: set[int],
    handle_resolver: Any,
    artifact: bool,
    projection_cache: dict[tuple[int, str], Any],
) -> Any:
    """Encode a non-container from explicit projections and/or public structural fields."""
    if artifact:
        try:
            projection = getattr(value, "artifact_data", None)
        except Exception as exc:
            raise TypeError(
                "AuthoringSnapshot could not read %s.artifact_data at %s" %
                (_qualified_type(value), path)) from exc
        if projection is not None:
            if not callable(projection):
                raise TypeError(
                    "AuthoringSnapshot expected %s.artifact_data to be callable at %s" %
                    (_qualified_type(value), path))
            projected = _call_projection(
                value, "artifact_data", projection, path, projection_cache)
            return {"$object": {
                "type": _qualified_type(value),
                "projections": {
                    "artifact_data": _canonical(
                        projected,
                        path="%s<%s>.artifact_data()" % (path, _qualified_type(value)),
                        active=active,
                        handle_resolver=handle_resolver,
                        artifact=True,
                        projection_cache=projection_cache,
                    ),
                },
            }}
    projections: dict[str, Any] = {}
    for accessor in ("to_data", "to_dict", "options"):
        try:
            member = getattr(value, accessor, None)
        except Exception as exc:  # property access is part of the structural protocol
            raise TypeError(
                "AuthoringSnapshot could not read %s.%s at %s" %
                (_qualified_type(value), accessor, path)) from exc
        if callable(member):
            projections[accessor] = _call_projection(
                value, accessor, member, path, projection_cache)
        elif member is not None:
            if accessor == "options" and isinstance(member, Mapping):
                projections[accessor] = member
            else:
                raise TypeError(
                    "AuthoringSnapshot expected %s.%s to be callable at %s" %
                    (_qualified_type(value), accessor, path))

    fields = _public_structural_fields(value, path)
    if fields:
        projections["fields"] = fields
    if not projections and not callable(value) and not _class_has_public_behavior(type(value)):
        raise TypeError(
            "AuthoringSnapshot cannot encode opaque %s at %s: expose to_data(), to_dict(), "
            "options(), or public structural fields" % (_qualified_type(value), path))
    implementation = _class_implementation_projection(
        type(value),
        path="%s<%s>.implementation" % (path, _qualified_type(value)),
        active=active,
        handle_resolver=handle_resolver,
        artifact=artifact,
        canonical=lambda item, **kwargs: _canonical(
            item, projection_cache=projection_cache, **kwargs),
        dependency_cache=projection_cache,
    )
    if implementation:
        projections["implementation"] = implementation
    return {"$object": {
        "type": _qualified_type(value),
        "projections": _canonical(
            projections,
            path="%s<%s>" % (path, _qualified_type(value)),
            active=active,
            handle_resolver=handle_resolver,
            artifact=artifact,
            projection_cache=projection_cache,
        ),
    }}


def _call_projection(
    value: Any,
    name: str,
    fn: Any,
    path: str,
    projection_cache: dict[tuple[int, str], Any],
) -> Any:
    """Call one structural projection without swallowing or replacing its exception."""
    key = (id(value), name)
    cached = projection_cache.get(key)
    if cached is not None:
        cached_value, cached_result = cached
        if cached_value is value:
            return cached_result
    try:
        result = fn()
    except Exception as exc:
        add_note = getattr(exc, "add_note", None)
        if callable(add_note):
            add_note("while AuthoringSnapshot called %s.%s() at %s" %
                     (_qualified_type(value), name, path))
        raise
    # Keep the projected object alive with its result.  Snapshot payload builders may create
    # short-lived canonical Handle copies; caching by a bare id lets CPython reuse that id for the
    # next copy and return another declaration's projection.
    projection_cache[key] = (value, result)
    return result


def _public_structural_fields(value: Any, path: str) -> dict[str, Any]:
    """Read stored Python state, or public native-extension data when no storage is exposed."""
    fields: dict[str, Any] = {}
    stored_names: set[str] = set()
    try:
        stored_names.update(vars(value))
    except TypeError:
        pass
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if isinstance(slot, str) and slot.startswith("__") and not slot.endswith("__"):
                slot = "_%s%s" % (cls.__name__.lstrip("_"), slot)
            stored_names.add(slot)
    stored_names.difference_update({"__dict__", "__weakref__", "_frozen", "_snapshot",
                                    "_canonical_json", "_hash", "_artifact_canonical_json",
                                    "_artifact_hash"})
    if stored_names:
        for name in sorted(stored_names):
            if not isinstance(name, str) or name.startswith("__"):
                continue
            try:
                fields[name] = getattr(value, name)
            except AttributeError:
                continue  # an unset optional slot carries no state
            except Exception as exc:
                raise TypeError("AuthoringSnapshot could not read %s.%s at %s" %
                                (_qualified_type(value), name, path)) from exc
        return fields

    # pybind/native records generally expose no ``__dict__`` or Python slots. Their documented
    # public, non-callable attributes are their only structural projection (e.g. ModelSpec).
    try:
        names = sorted(
            name for name in dir(value) if isinstance(name, str) and not name.startswith("_"))
    except Exception as exc:
        raise TypeError("AuthoringSnapshot could not inspect %s at %s" %
                        (_qualified_type(value), path)) from exc
    for name in names:
        if name in ("to_data", "to_dict", "options", "frozen"):
            continue
        try:
            member = getattr(value, name)
        except Exception as exc:
            raise TypeError("AuthoringSnapshot could not read %s.%s at %s" %
                            (_qualified_type(value), name, path)) from exc
        if not callable(member):
            fields[name] = member
    return fields


def _qualified_type(value: Any) -> str:
    cls = type(value)
    cls = getattr(cls, "_pops_unfrozen_type", cls)
    return "%s.%s" % (cls.__module__, cls.__qualname__)


__all__ = ["_canonical"]
