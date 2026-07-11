"""Strict canonical projections used by :mod:`pops.problem._snapshot`.

This module contains the array-free, authenticated conversion of authoring values into stable
JSON-ready data.  It is deliberately independent from snapshot lifecycle and hashing.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from decimal import Decimal
from fractions import Fraction
from typing import Any
from urllib.parse import quote

from pops.ir.literals import ScalarLiteral
from pops.model.handles import (
    Handle,
    OperatorHandle as ModelOperatorHandle,
    OwnerPath,
    ParamHandle,
)


def _canonical(
    value: Any,
    *,
    path: str = "$",
    active: set[int] | None = None,
    handle_resolver: Any = None,
    artifact: bool = False,
) -> Any:
    """Return a strict, deterministic JSON view of ``value``.

    No lossy fallback exists.  An object is encoded from structural projections and its qualified
    Python type, or the snapshot fails at the exact path.  ``active`` detects reference cycles while
    still allowing the same immutable descriptor to appear in two independent branches.
    """
    if active is None:
        active = set()
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return {"$scalar": {"kind": "integer", "value": str(value)}}
    if isinstance(value, Fraction):
        return {"$scalar": {"kind": "rational", "numerator": str(value.numerator),
                            "denominator": str(value.denominator)}}
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("ProblemSnapshot cannot encode a non-finite Decimal")
        return {"$scalar": {"kind": "decimal", "value": str(value)}}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("ProblemSnapshot cannot encode a non-finite float at %s" % path)
        return {"$scalar": {"kind": "binary64", "value": value.hex()}}
    if isinstance(value, str):
        return value
    marker = id(value)
    if marker in active:
        raise ValueError("ProblemSnapshot cannot encode a reference cycle at %s" % path)
    active.add(marker)
    try:
        return _canonical_compound(
            value, path=path, active=active, handle_resolver=handle_resolver,
            artifact=artifact)
    finally:
        active.remove(marker)


def _canonical_compound(
    value: Any,
    *,
    path: str,
    active: set[int],
    handle_resolver: Any,
    artifact: bool,
) -> Any:
    """Canonicalise a container, handle, literal or structural Python object."""
    if type(value).__module__.split(".", 1)[0] == "numpy":
        raise TypeError(
            "ProblemSnapshot cannot encode numpy runtime data at %s; declare metadata, not arrays"
            % path)
    if isinstance(value, (list, tuple)):
        return [_canonical(
                    item, path="%s[%d]" % (path, index), active=active,
                    handle_resolver=handle_resolver, artifact=artifact)
                for index, item in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        items = [_canonical(
                    item, path="%s{item}" % path, active=active,
                    handle_resolver=handle_resolver, artifact=artifact)
                 for item in value]
        return {"$set": sorted(items, key=lambda item: json.dumps(
            item, sort_keys=True, separators=(",", ":"), allow_nan=False))}
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: _canonical(
                        item, path="%s.%s" % (path, key), active=active,
                        handle_resolver=handle_resolver, artifact=artifact)
                    for key, item in value.items()}
        entries = [
            (_canonical(
                key, path="%s{key}" % path, active=active,
                handle_resolver=handle_resolver, artifact=artifact),
             _canonical(
                 item, path="%s{%r}" % (path, key), active=active,
                 handle_resolver=handle_resolver, artifact=artifact))
            for key, item in value.items()
        ]
        entries.sort(key=lambda pair: json.dumps(
            pair[0], sort_keys=True, separators=(",", ":"), allow_nan=False))
        return {"$map": [[key, item] for key, item in entries]}
    if isinstance(value, OwnerPath):
        canonical_owner = value.canonical()
        data = _call_projection(canonical_owner, "to_data", canonical_owner.to_data, path)
        if OwnerPath.from_data(data) != canonical_owner:
            raise ValueError("OwnerPath.to_data() does not round-trip at %s" % path)
        return {"$owner_path": data}
    if isinstance(value, Handle):
        return {"$handle": _canonical_handle_identity(
            value, path, handle_resolver=handle_resolver)}
    if isinstance(value, ScalarLiteral):
        data = _call_projection(value, "to_data", value.to_data, path)
        return {"$scalar": _canonical_literal_data(data, path="%s.to_data()" % path)}
    return _canonical_object(
        value, path=path, active=active, handle_resolver=handle_resolver,
        artifact=artifact)


def _canonical_handle_identity(
    value: Handle,
    path: str,
    *,
    handle_resolver: Any = None,
) -> dict[str, Any]:
    """Validate an authenticated Handle projection without lossy coercion or duck typing."""
    if callable(handle_resolver):
        authored = value
        resolved = handle_resolver(value)
        if not isinstance(resolved, Handle) or not resolved.is_resolved:
            raise TypeError(
                "ProblemSnapshot handle resolver must return a canonical Handle at %s" % path)
        if (resolved.schema_version, resolved.kind, resolved.local_id) != (
                authored.schema_version, authored.kind, authored.local_id):
            raise ValueError(
                "ProblemSnapshot handle resolver changed declaration identity at %s" % path)
        if isinstance(authored, ModelOperatorHandle) and (
                not isinstance(resolved, ModelOperatorHandle)
                or resolved.registered_operator_name != authored.registered_operator_name):
            raise ValueError(
                "ProblemSnapshot handle resolver changed operator target at %s" % path)
        if isinstance(authored, ParamHandle) and (
                not isinstance(resolved, ParamHandle)
                or resolved.param_kind != authored.param_kind):
            raise ValueError(
                "ProblemSnapshot handle resolver changed parameter kind at %s" % path)
        if authored.is_resolved \
                and resolved.canonical_identity() != authored.canonical_identity():
            raise ValueError(
                "ProblemSnapshot handle resolver changed an already canonical identity at %s"
                % path)
        value = resolved
    elif not value.is_resolved:
        raise TypeError(
            "ProblemSnapshot cannot serialize unresolved handle %s at %s without an "
            "authoritative resolver" % (value.qualified_id, path))
    identity = _call_projection(value, "canonical_identity", value.canonical_identity, path)
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
) -> Any:
    """Encode a non-container from explicit projections and/or public structural fields."""
    if artifact:
        try:
            projection = getattr(value, "artifact_data", None)
        except Exception as exc:
            raise TypeError(
                "ProblemSnapshot could not read %s.artifact_data at %s" %
                (_qualified_type(value), path)) from exc
        if projection is not None:
            if not callable(projection):
                raise TypeError(
                    "ProblemSnapshot expected %s.artifact_data to be callable at %s" %
                    (_qualified_type(value), path))
            projected = _call_projection(value, "artifact_data", projection, path)
            return {"$object": {
                "type": _qualified_type(value),
                "projections": {
                    "artifact_data": _canonical(
                        projected,
                        path="%s<%s>.artifact_data()" % (path, _qualified_type(value)),
                        active=active,
                        handle_resolver=handle_resolver,
                        artifact=True,
                    ),
                },
            }}
    projections: dict[str, Any] = {}
    for accessor in ("to_data", "to_dict", "options"):
        try:
            member = getattr(value, accessor, None)
        except Exception as exc:  # property access is part of the structural protocol
            raise TypeError(
                "ProblemSnapshot could not read %s.%s at %s" %
                (_qualified_type(value), accessor, path)) from exc
        if callable(member):
            projections[accessor] = _call_projection(value, accessor, member, path)
        elif member is not None:
            if accessor == "options" and isinstance(member, Mapping):
                projections[accessor] = member
            else:
                raise TypeError(
                    "ProblemSnapshot expected %s.%s to be callable at %s" %
                    (_qualified_type(value), accessor, path))

    fields = _public_structural_fields(value, path)
    if fields:
        projections["fields"] = fields
    if not projections:
        raise TypeError(
            "ProblemSnapshot cannot encode opaque %s at %s: expose to_data(), to_dict(), "
            "options(), or public structural fields" % (_qualified_type(value), path))
    return {"$object": {
        "type": _qualified_type(value),
        "projections": _canonical(
            projections,
            path="%s<%s>" % (path, _qualified_type(value)),
            active=active,
            handle_resolver=handle_resolver,
            artifact=artifact,
        ),
    }}


def _call_projection(value: Any, name: str, fn: Any, path: str) -> Any:
    """Call one structural projection without swallowing or replacing its exception."""
    try:
        return fn()
    except Exception as exc:
        if hasattr(exc, "add_note"):
            exc.add_note("while ProblemSnapshot called %s.%s() at %s" %
                         (_qualified_type(value), name, path))
        raise


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
                raise TypeError("ProblemSnapshot could not read %s.%s at %s" %
                                (_qualified_type(value), name, path)) from exc
        return fields

    # pybind/native records generally expose no ``__dict__`` or Python slots. Their documented
    # public, non-callable attributes are their only structural projection (e.g. ModelSpec).
    try:
        names = sorted(
            name for name in dir(value) if isinstance(name, str) and not name.startswith("_"))
    except Exception as exc:
        raise TypeError("ProblemSnapshot could not inspect %s at %s" %
                        (_qualified_type(value), path)) from exc
    for name in names:
        if name in ("to_data", "to_dict", "options", "frozen"):
            continue
        try:
            member = getattr(value, name)
        except Exception as exc:
            raise TypeError("ProblemSnapshot could not read %s.%s at %s" %
                            (_qualified_type(value), name, path)) from exc
        if not callable(member):
            fields[name] = member
    return fields


def _qualified_type(value: Any) -> str:
    cls = type(value)
    cls = getattr(cls, "_pops_unfrozen_type", cls)
    return "%s.%s" % (cls.__module__, cls.__qualname__)


def _canonical_literal_data(data: Any, *, path: str) -> Any:
    """Canonicalize an already JSON-shaped ScalarLiteral view without re-tagging its integers."""
    if isinstance(data, dict):
        if not all(isinstance(key, str) for key in data):
            raise TypeError("ScalarLiteral.to_data() requires string keys at %s" % path)
        return {key: _canonical_literal_data(item, path="%s.%s" % (path, key))
                for key, item in data.items()}
    if isinstance(data, (list, tuple)):
        return [_canonical_literal_data(item, path="%s[%d]" % (path, index))
                for index, item in enumerate(data)]
    if isinstance(data, float) and not math.isfinite(data):
        raise ValueError("ScalarLiteral.to_data() contains a non-finite float at %s" % path)
    if data is None or isinstance(data, (bool, int, float, str)):
        return data
    raise TypeError("ScalarLiteral.to_data() is not JSON-ready at %s (got %s)" %
                    (path, _qualified_type(data)))


__all__ = ["_canonical"]
