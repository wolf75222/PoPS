"""pops.problem._snapshot -- the frozen ProblemSnapshot the compile cache keys on (ADC-563).

``Problem.freeze()`` returns a :class:`ProblemSnapshot`: an inert, JSON-ready, array-free capture of
the whole assembly (blocks / fields / params / aux / outputs / constraints / time / layout) with a
stable ``.hash`` (sha256 over the canonical ``to_dict``). ``pops.compile`` freezes the Problem
before invoking the compiler and passes ``snapshot.hash`` into the real cache lookup, so a
post-compile mutation cannot change a bound artifact -- the snapshot is the FROZEN identity the
compile stream keys on.

The snapshot holds PLAIN values only: no runtime object, no numpy array, no live descriptor. Objects
must expose structural data (``to_data`` / ``to_dict`` / ``options`` and/or public fields); an
opaque value is rejected instead of being collapsed to its class name.  Every object projection is
qualified by its Python type, recursively canonicalised and deeply detached.  It imports only the
pure-Python PoPS literal/handle value types used to authenticate special encodings, never ``_pops``
or the runtime.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from decimal import Decimal
from fractions import Fraction
from typing import Any
from urllib.parse import quote

from pops.ir.literals import ScalarLiteral
from pops.model.handles import Handle, OperatorHandle as ModelOperatorHandle, OwnerPath

#: Bumped only when the snapshot's canonical shape changes (a field rename / removal); an additive
#: field keeps version 1 so an old hash and a new hash of the SAME assembly stay comparable.
SNAPSHOT_SCHEMA_VERSION = 4


def _canonical(
    value: Any,
    *,
    path: str = "$",
    active: set[int] | None = None,
    handle_resolver: Any = None,
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
            value, path=path, active=active, handle_resolver=handle_resolver)
    finally:
        active.remove(marker)


def _canonical_compound(
    value: Any,
    *,
    path: str,
    active: set[int],
    handle_resolver: Any,
) -> Any:
    """Canonicalise a container, handle, literal or structural Python object."""
    if type(value).__module__.split(".", 1)[0] == "numpy":
        raise TypeError(
            "ProblemSnapshot cannot encode numpy runtime data at %s; declare metadata, not arrays"
            % path)
    if isinstance(value, (list, tuple)):
        return [_canonical(
                    item, path="%s[%d]" % (path, index), active=active,
                    handle_resolver=handle_resolver)
                for index, item in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        items = [_canonical(
                    item, path="%s{item}" % path, active=active,
                    handle_resolver=handle_resolver)
                 for item in value]
        return {"$set": sorted(items, key=lambda item: json.dumps(
            item, sort_keys=True, separators=(",", ":"), allow_nan=False))}
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: _canonical(
                        item, path="%s.%s" % (path, key), active=active,
                        handle_resolver=handle_resolver)
                    for key, item in value.items()}
        entries = [
            (_canonical(
                key, path="%s{key}" % path, active=active,
                handle_resolver=handle_resolver),
             _canonical(
                 item, path="%s{%r}" % (path, key), active=active,
                 handle_resolver=handle_resolver))
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
        value, path=path, active=active, handle_resolver=handle_resolver)


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
) -> Any:
    """Encode a non-container from explicit projections and/or public structural fields."""
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
                                    "_canonical_json", "_hash"})
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
        names = sorted(name for name in dir(value) if isinstance(name, str) and not name.startswith("_"))
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


class ProblemSnapshot:
    """The frozen, JSON-ready capture of a :class:`~pops.problem.problem.Problem` (ADC-563).

    A plain inert value: :attr:`to_dict` is the canonical dict (deep, array-free, no runtime object)
    and :attr:`hash` is its stable sha256. Two snapshots of the same assembly have the same hash; a
    mutation before freeze changes it, a mutation after freeze is impossible (the Problem raises).
    ``pops.compile`` attaches it to the compiled handle (``compiled._problem_snapshot``) after the
    compile driver has included :attr:`hash` in the artifact hash, cache key, path and sidecar.
    """

    schema_version = SNAPSHOT_SCHEMA_VERSION
    __slots__ = ("_canonical_json", "_hash")

    def __init__(self, payload: Any, *, handle_resolver: Any = None) -> None:
        # A deep, canonical, JSON-ready copy: no shared reference to a live registry, no runtime
        # object -- so there is no shallow-copy escape from the frozen identity.
        canonical_payload = _canonical(payload, handle_resolver=handle_resolver)
        if not isinstance(canonical_payload, dict):
            raise TypeError("ProblemSnapshot payload must be a mapping")
        if "schema_version" in canonical_payload:
            raise ValueError("ProblemSnapshot payload cannot define reserved key 'schema_version'")
        out = dict(canonical_payload)
        out["schema_version"] = self.schema_version
        canonical_json = json.dumps(
            out, sort_keys=True, separators=(",", ":"), allow_nan=False)
        object.__setattr__(self, "_canonical_json", canonical_json)
        object.__setattr__(
            self, "_hash", hashlib.sha256(canonical_json.encode("utf-8")).hexdigest())

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("ProblemSnapshot is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("ProblemSnapshot is immutable")

    def to_dict(self) -> Any:
        """The canonical, JSON-ready dict of the frozen assembly (stamped with the schema version)."""
        # Decode afresh so callers receive an ordinary JSON-ready deep copy with no route back into
        # the frozen cache identity.
        return json.loads(self._canonical_json)

    @property
    def hash(self) -> Any:
        """The stable sha256 (64-hex) over the canonical ``to_dict`` (computed once, then cached)."""
        return self._hash

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, ProblemSnapshot) and self.hash == other.hash

    def __hash__(self) -> int:
        return hash(self.hash)

    def __repr__(self) -> str:
        return "ProblemSnapshot(hash=%s...)" % self.hash[:12]


def build_problem_snapshot(problem: Any) -> Any:
    """Build the :class:`ProblemSnapshot` of @p problem (the frozen input to the compile cache key).

    Reads the Problem's raw typed registries (not its intentionally concise inspection view) and
    canonicalises them into a JSON-ready, deep, inert payload. It computes nothing on a grid and
    imports no ``_pops``; the resulting ``.hash`` is an input to the driver-owned artifact hash,
    path and cache sidecar so a mutated Problem cannot silently rebind a compiled artifact."""
    from pops.problem._snapshot_payload import problem_snapshot_payload

    payload = problem_snapshot_payload(problem)
    return ProblemSnapshot(payload, handle_resolver=problem.resolve)


def validate_problem_snapshot(snapshot: Any) -> str:
    """Return the authenticated 64-hex hash of an exact :class:`ProblemSnapshot` value."""
    if type(snapshot) is not ProblemSnapshot:
        raise TypeError(
            "problem_snapshot must be a pops.problem.ProblemSnapshot, not %r"
            % type(snapshot).__name__)
    snapshot_hash = snapshot.hash
    if not isinstance(snapshot_hash, str) or len(snapshot_hash) != 64 \
            or any(char not in "0123456789abcdef" for char in snapshot_hash):
        raise ValueError("ProblemSnapshot.hash must be exactly 64 lowercase hexadecimal characters")
    canonical_json = json.dumps(
        snapshot.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    expected = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    if snapshot_hash != expected:
        raise ValueError("ProblemSnapshot.hash does not match its canonical payload")
    return snapshot_hash


def prepare_compile_snapshot(problem: Any, time: Any) -> ProblemSnapshot:
    """Freeze the public authoring graph before cache lookup and return its validated snapshot."""
    freeze = getattr(problem, "freeze", None)
    if not callable(freeze):
        raise TypeError("pops.compile requires a Problem exposing freeze()")
    snapshot = freeze()
    validate_problem_snapshot(snapshot)
    time_freeze = getattr(time, "freeze", None) if time is not None else None
    if callable(time_freeze):
        time_freeze()
    return snapshot


def attach_problem_snapshot(compiled: Any, snapshot: Any) -> None:
    """Attach an already-keyed snapshot and seal the completed public artifact."""
    validate_problem_snapshot(snapshot)
    existing = getattr(compiled, "_problem_snapshot", None)
    if existing is not None and existing is not snapshot:
        raise ValueError("compiled artifact carries a different ProblemSnapshot than its cache key")
    if existing is None:
        compiled._problem_snapshot = snapshot
    if hasattr(compiled, "_seal"):
        compiled._seal()


__all__ = [
    "ProblemSnapshot", "build_problem_snapshot", "validate_problem_snapshot",
    "prepare_compile_snapshot", "attach_problem_snapshot", "SNAPSHOT_SCHEMA_VERSION",
]
