"""Frozen authoring snapshots and their reproducibility/compile identities.

``Case.freeze()`` returns an :class:`AuthoringSnapshot`: an inert, JSON-ready, array-free capture
of the assembly.  ``pops.compile`` widens that capture with the *effective* layout, time program and
external libraries.  This distinction matters because those values may be supplied directly to
``pops.compile`` and therefore do not necessarily live on ``Case``.  The compiler snapshot is the
one authority used for cache identity and retained provenance; no effective compile input may live
only in a side attribute on the compiled handle.

Canonical projection is isolated in :mod:`pops.problem._snapshot_canonical`; this module owns only
the immutable value, hashing, construction, validation, and attachment lifecycle.
"""
from __future__ import annotations

from pops._structural_data import canonical_json as _encode_json, json_containers
from typing import Any
import weakref
from types import FunctionType

import pops.identity.semantic as _semantic_module

from pops.identity import Identity, canonical_sha256
from pops.identity.semantic import semantic_identity, semantic_value
from pops.model.ownership import _definition_fingerprint_scope
from pops.problem._snapshot_canonical import _canonical

#: Bumped when the full snapshot's canonical shape changes. The compile-only projection has its own
#: version below, so artifact-identity evolution does not rewrite the reproducibility schema.
SNAPSHOT_SCHEMA_VERSION = 11

#: Independent namespace for the compile-identity projection. Changing which declaration facts
#: affect generated artifacts bumps this version without rewriting the full snapshot schema.
ARTIFACT_SCHEMA_VERSION = 6


class AuthoringSnapshot:
    """The frozen, JSON-ready capture of one complete PoPS authoring transaction.

    A plain inert value: :attr:`to_dict` is the canonical dict (deep, array-free, no runtime object)
    and :attr:`hash` is its stable sha256. Two snapshots of the same assembly have the same hash; a
    mutation before freeze changes it, a mutation after freeze is impossible (the Case raises).
    ``pops.resolve`` captures it in the immutable simulation plan; compilation includes
    :attr:`artifact_hash` in the artifact hash, cache key, path and sidecar. :attr:`hash` remains the
    exact authored/reproducibility identity.
    """

    schema_version = SNAPSHOT_SCHEMA_VERSION
    __slots__ = (
        "_canonical_json", "_hash", "_semantic_canonical_json", "_semantic_identity",
        "_artifact_canonical_json", "_artifact_hash", "__weakref__",
    )

    def __init__(
        self,
        payload: Any,
        *,
        handle_resolver: Any = None,
        artifact_payload: Any = None,
        semantic_payload: Any = None,
    ) -> None:
        # A deep, canonical, JSON-ready copy: no shared reference to a live registry, no runtime
        # object -- so there is no shallow-copy escape from the frozen identity.
        projection_cache: dict[tuple[int, str], Any] = {}
        canonical_payload = _canonical(
            payload, handle_resolver=handle_resolver, projection_cache=projection_cache)
        if not isinstance(canonical_payload, dict):
            raise TypeError("AuthoringSnapshot payload must be a mapping")
        if "schema_version" in canonical_payload:
            raise ValueError("AuthoringSnapshot payload cannot define reserved key 'schema_version'")
        out = dict(canonical_payload)
        out["schema_version"] = self.schema_version
        canonical_json = _encode_json(
            out, sort_keys=False, allow_nan=False)
        object.__setattr__(self, "_canonical_json", canonical_json)
        full_hash = canonical_sha256(out)
        object.__setattr__(self, "_hash", full_hash)

        # Case builders always supply their closed scientific projection. Direct construction is
        # a low-level provenance seam: there the caller's explicit payload is also the semantic
        # declaration, after the strict snapshot projector has made it an inert value.
        semantic_source = canonical_payload if semantic_payload is None else semantic_payload
        semantic_data = semantic_value(semantic_source, where="AuthoringSnapshot semantic payload")
        semantic_json = _encode_json(
            semantic_data, sort_keys=True, allow_nan=False)
        object.__setattr__(self, "_semantic_canonical_json", semantic_json)
        semantic_key = semantic_identity(semantic_data)
        object.__setattr__(self, "_semantic_identity", semantic_key)

        # This is a separately versioned preimage, not a scrub of the full canonical dict. The
        # Case builder supplies an explicit parameter/model projection; standalone snapshots use
        # the same raw payload and let objects opt in through artifact_data().
        artifact_source = payload if artifact_payload is None else artifact_payload
        canonical_artifact_payload = _canonical(
            artifact_source, handle_resolver=handle_resolver, artifact=True,
            projection_cache=projection_cache)
        if not isinstance(canonical_artifact_payload, dict):
            raise TypeError("AuthoringSnapshot artifact payload must be a mapping")
        if "schema_version" in canonical_artifact_payload:
            raise ValueError(
                "AuthoringSnapshot artifact payload cannot define reserved key 'schema_version'")
        artifact_envelope = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "problem_snapshot_schema_version": self.schema_version,
            "payload": canonical_artifact_payload,
        }
        artifact_json = _encode_json(
            artifact_envelope, sort_keys=False, allow_nan=False)
        object.__setattr__(self, "_artifact_canonical_json", artifact_json)
        artifact_hash = canonical_sha256(artifact_envelope)
        object.__setattr__(self, "_artifact_hash", artifact_hash)
        # Only the closed builtin JSON language has an exact JSON decode preimage.
        # Subclasses, tuples, non-string keys and floats must take the
        # complete validator; user hooks never establish a construction receipt.
        if all(_plain_json_data(value) for value in (out, semantic_data, artifact_envelope)):
            _remember_snapshot(self, (
                canonical_json, full_hash, semantic_json,
                semantic_key.domain, semantic_key.schema_version,
                semantic_key.algorithm, semantic_key.digest,
                artifact_json, artifact_hash,
            ))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("AuthoringSnapshot is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("AuthoringSnapshot is immutable")

    def to_dict(self) -> Any:
        """The canonical, JSON-ready dict of the frozen assembly (stamped with the schema version)."""
        # Decode afresh so callers receive an ordinary JSON-ready deep copy with no route back into
        # the frozen cache identity.
        return json_containers(self._canonical_json)

    @property
    def hash(self) -> Any:
        """Provenance sha256 over canonical PoPS bytes of the exact authoring capture."""
        return self._hash

    @property
    def semantic_identity(self) -> Identity:
        """Scientific identity, independent from presentation, runtime and lowering choices."""
        return self._semantic_identity

    def semantic_to_dict(self) -> dict[str, Any]:
        """Return a detached copy of the exact semantic-identity payload."""
        return json_containers(self._semantic_canonical_json)

    @property
    def artifact_hash(self) -> str:
        """Stable sha256 of compile-relevant data, excluding runtime bind values/defaults.

        This still covers kind, dtype, unit, domain, storage, phase/invalidation, constant values,
        ownership, and every non-parameter artifact input.
        """
        return self._artifact_hash

    def artifact_to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-ready copy of the versioned artifact projection."""
        return json_containers(self._artifact_canonical_json)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, AuthoringSnapshot) and self.hash == other.hash

    def __hash__(self) -> int:
        return hash(self.hash)

    def __repr__(self) -> str:
        return "AuthoringSnapshot(hash=%s..., semantic_identity=%s..., artifact_hash=%s...)" % (
            self.hash[:12], self.semantic_identity.hexdigest[:12], self.artifact_hash[:12])


# An identity-keyed weak table is deliberate: AuthoringSnapshot.__hash__/__eq__
# use authored hashes, which object.__setattr__ can replace. No table key or
# receipt retains a snapshot or its mutable Identity instance.
_SNAPSHOT_RECEIPTS: dict[int, tuple[Any, tuple[Any, ...], tuple[int, ...]]] = {}
_SNAPSHOT_SLOTS = (
    "_canonical_json", "_hash", "_semantic_canonical_json", "_semantic_identity",
    "_artifact_canonical_json", "_artifact_hash",
)
_SNAPSHOT_HOOKS = (*_SNAPSHOT_SLOTS, "to_dict", "semantic_to_dict", "artifact_to_dict",
                   "hash", "semantic_identity", "artifact_hash", "__getattribute__")
_IDENTITY_SLOTS = ("domain", "schema_version", "algorithm", "digest")
_IDENTITY_HOOKS = (*_IDENTITY_SLOTS, "__getattribute__", "__eq__")


def _implementation_tokens(values: Any) -> tuple[Any, ...]:
    tokens = []
    for value in values:
        tokens.append(value)
        if type(value) is FunctionType:
            tokens.append(value.__code__)
        elif type(value) is property:
            for accessor in (value.fget, value.fset, value.fdel):
                tokens.append(accessor)
                if type(accessor) is FunctionType:
                    tokens.append(accessor.__code__)
    return tuple(tokens)


def _snapshot_protocol_tokens() -> tuple[Any, ...]:
    return _implementation_tokens((
        *(AuthoringSnapshot.__dict__.get(name) for name in _SNAPSHOT_HOOKS),
        *(Identity.__dict__.get(name) for name in _IDENTITY_HOOKS),
        _encode_json, json_containers, canonical_sha256, semantic_identity,
        _semantic_module.semantic_value, _semantic_module.make_identity,
    ))


_SNAPSHOT_PROTOCOL = _snapshot_protocol_tokens()


def _plain_json_data(value: Any) -> bool:
    """Prove the closed JSON/CBOR intersection without user protocols.

    Binary64 values are already exact scalar records at this boundary: the
    canonical CBOR codec deliberately refuses raw floats, including signed zero.
    """
    active: set[int] = set()
    complete: set[int] = set()
    work = [(False, value)]
    while work:
        leaving, item = work.pop()
        if leaving:
            active.remove(id(item))
            complete.add(id(item))
            continue
        kind = type(item)
        if item is None or kind in (str, int, bool):
            continue
        if kind not in (dict, list):
            return False
        key = id(item)
        if key in active:
            return False
        if key in complete:
            continue
        active.add(key)
        work.append((True, item))
        if kind is dict:
            if any(type(key) is not str for key in item):
                return False
            work.extend((False, child) for child in item.values())
        else:
            work.extend((False, child) for child in item)
    return True


def _snapshot_receipt_values(snapshot: Any) -> tuple[Any, ...] | None:
    if type(snapshot) is not AuthoringSnapshot:
        return None
    tokens = _snapshot_protocol_tokens()
    if len(tokens) != len(_SNAPSHOT_PROTOCOL) or any(
            actual is not expected for actual, expected in zip(tokens, _SNAPSHOT_PROTOCOL, strict=True)):
        return None
    values = tuple(object.__getattribute__(snapshot, name) for name in _SNAPSHOT_SLOTS)
    semantic = values[3]
    if type(semantic) is not Identity:
        return None
    parts = tuple(object.__getattribute__(semantic, name) for name in _IDENTITY_SLOTS)
    result = (*values[:3], *parts, *values[4:])
    expected_types = (str, str, str, str, int, str, bytes, str, str)
    if any(type(value) is not kind for value, kind in zip(result, expected_types, strict=True)):
        return None
    return result


def _snapshot_receipt_versions() -> tuple[int, ...] | None:
    values = (AuthoringSnapshot.schema_version, SNAPSHOT_SCHEMA_VERSION,
              ARTIFACT_SCHEMA_VERSION, _semantic_module.SEMANTIC_SCHEMA_VERSION)
    return values if all(type(value) is int for value in values) else None


def _remember_snapshot(snapshot: Any, expected: tuple[Any, ...]) -> None:
    current = _snapshot_receipt_values(snapshot)
    versions = _snapshot_receipt_versions()
    if current is None or versions is None or current != expected:
        return
    key = id(snapshot)

    def release(reference: Any) -> None:
        entry = _SNAPSHOT_RECEIPTS.get(key)
        if entry is not None and entry[0] is reference:
            del _SNAPSHOT_RECEIPTS[key]

    _SNAPSHOT_RECEIPTS[key] = (weakref.ref(snapshot, release), expected, versions)


def _has_exact_snapshot_receipt(snapshot: Any) -> bool:
    entry = _SNAPSHOT_RECEIPTS.get(id(snapshot))
    if entry is None or entry[0]() is not snapshot:
        return False
    # Every byte and scalar is compared, including Identity fields copied by
    # value. Object identity only selects the receipt; it is never its verdict.
    current = _snapshot_receipt_values(snapshot)
    versions = _snapshot_receipt_versions()
    return current is not None and versions is not None and current == entry[1] and versions == entry[2]


@_definition_fingerprint_scope()
def build_problem_snapshot(problem: Any) -> Any:
    """Build the :class:`AuthoringSnapshot` of @p problem (the frozen input to the compile cache key).

    Reads the Case's raw typed registries (not its intentionally concise inspection view) and
    canonicalises them into a JSON-ready, deep, inert payload. It computes nothing on a grid and
    imports no ``_pops``. ``.hash`` preserves the exact authored capture and ``.artifact_hash`` is
    the driver-owned compile identity, so a runtime default can change without recompiling while a
    constant or ABI-relevant declaration change cannot silently reuse an artifact."""
    # A successful freeze is a two-phase commit: the snapshot is captured before descriptors are
    # sealed, then containers may be replaced by immutable equivalents (list -> tuple,
    # dict -> mappingproxy).  Re-canonicalising those lifecycle representations would manufacture a
    # different authoring identity for the same frozen Case.  The committed snapshot is therefore
    # the sole authority once frozen.
    if getattr(problem, "frozen", False):
        snapshot = getattr(problem, "snapshot", None)
        if type(snapshot) is not AuthoringSnapshot:
            raise RuntimeError("frozen Case has no committed AuthoringSnapshot")
        return snapshot

    from pops.problem._snapshot_payload import (
        problem_semantic_payload,
        problem_snapshot_artifact_payload,
        problem_snapshot_payload,
    )

    payload = problem_snapshot_payload(problem)
    artifact_payload = problem_snapshot_artifact_payload(problem)
    semantic_payload = problem_semantic_payload(
        problem, layout=None, time=problem._time_registry.program)
    return AuthoringSnapshot(
        payload,
        handle_resolver=problem.resolve,
        artifact_payload=artifact_payload,
        semantic_payload=semantic_payload,
    )


@_definition_fingerprint_scope()
def build_authoring_snapshot(
    problem: Any,
    *,
    layout: Any,
    time: Any,
    libraries: Any = (),
) -> AuthoringSnapshot:
    """Capture every effective input selected by :func:`pops.compile`.

    ``layout`` and ``time`` are deliberately required keyword arguments.  Passing ``None`` is a
    meaningful value on routes that support it; omitting either would recreate the old split
    authority where an explicit compile argument was absent from the snapshot.  Library objects are
    projected structurally, so implementation/content hashes and manifests participate without
    retaining the live objects.
    """
    from pops.problem._snapshot_payload import (
        problem_semantic_payload,
        problem_snapshot_artifact_payload,
        problem_snapshot_payload,
    )

    full = problem_snapshot_payload(problem)
    artifact = problem_snapshot_artifact_payload(problem)
    semantic = problem_semantic_payload(problem, layout=layout, time=time)
    context = {
        "layout": layout,
        "time": _compile_time_snapshot_value(time),
        "libraries": tuple(libraries or ()),
    }
    full["compile_context"] = context
    artifact["compile_context"] = context
    def resolve_compile_handle(value: Any) -> Any:
        from pops.mesh import LayoutHandle, LayoutPlan

        if isinstance(layout, LayoutPlan) and isinstance(value, LayoutHandle):
            matches = [row.handle for row in layout.layouts if row.handle == value]
            if len(matches) != 1:
                raise ValueError(
                    "compile snapshot references a layout handle absent from its LayoutPlan")
            return matches[0]
        return problem.resolve(value)

    return AuthoringSnapshot(
        full,
        handle_resolver=resolve_compile_handle,
        artifact_payload=artifact,
        semantic_payload=semantic,
    )


def _compile_time_snapshot_value(program: Any) -> Any:
    """Return the lossless structural Program view used by the complete snapshot."""
    if program is None:
        return None
    serialize = getattr(program, "_serialize", None)
    if callable(serialize):
        return {
            "type": "%s.%s" % (type(program).__module__, type(program).__qualname__),
            "ir": serialize(),
            "hash": program._ir_hash() if callable(getattr(program, "_ir_hash", None)) else None,
        }
    return program


def validate_problem_snapshot(snapshot: Any) -> str:
    """Validate both identities and return the full reproducibility hash.

    The compile driver reads :attr:`AuthoringSnapshot.artifact_hash` only after this function has
    authenticated the exact full snapshot and separately versioned artifact projection.
    """
    if type(snapshot) is not AuthoringSnapshot:
        raise TypeError(
            "problem_snapshot must be a pops.problem.AuthoringSnapshot, not %r"
            % type(snapshot).__name__)
    if _has_exact_snapshot_receipt(snapshot):
        return snapshot.hash
    return _validate_problem_snapshot_full(snapshot)


def _validate_problem_snapshot_full(snapshot: AuthoringSnapshot) -> str:
    snapshot_hash = snapshot.hash
    if not isinstance(snapshot_hash, str) or len(snapshot_hash) != 64 \
            or any(char not in "0123456789abcdef" for char in snapshot_hash):
        raise ValueError("AuthoringSnapshot.hash must be exactly 64 lowercase hexadecimal characters")
    # The decoder already returns the exact builtin JSON value language. An
    # encode/decode roundtrip cannot add authority and copies deep snapshots
    # twice before hashing the same CBOR preimage.
    canonical_data = snapshot.to_dict()
    if not _plain_json_data(canonical_data):
        canonical_data = json_containers(_encode_json(canonical_data, sort_keys=False, allow_nan=False))
    expected = canonical_sha256(canonical_data)
    if snapshot_hash != expected:
        raise ValueError("AuthoringSnapshot.hash does not match its canonical payload")
    semantic = snapshot.semantic_identity
    if not isinstance(semantic, Identity) or semantic.domain != "semantic":
        raise TypeError("AuthoringSnapshot.semantic_identity must be a semantic Identity")
    semantic_data = snapshot.semantic_to_dict()
    if not _plain_json_data(semantic_data):
        semantic_data = json_containers(_encode_json(semantic_data, sort_keys=False, allow_nan=False))
    expected_semantic = semantic_identity(semantic_data)
    if semantic != expected_semantic:
        raise ValueError("AuthoringSnapshot.semantic_identity does not match its semantic payload")
    artifact_hash = snapshot.artifact_hash
    if not isinstance(artifact_hash, str) or len(artifact_hash) != 64 \
            or any(char not in "0123456789abcdef" for char in artifact_hash):
        raise ValueError(
            "AuthoringSnapshot.artifact_hash must be exactly 64 lowercase hexadecimal characters")
    artifact_data = snapshot.artifact_to_dict()
    if not _plain_json_data(artifact_data):
        artifact_data = json_containers(_encode_json(artifact_data, sort_keys=False, allow_nan=False))
    if not isinstance(artifact_data, dict) or set(artifact_data) != {
            "artifact_schema_version", "problem_snapshot_schema_version", "payload"}:
        raise ValueError("AuthoringSnapshot artifact projection has an invalid envelope")
    if artifact_data["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("AuthoringSnapshot artifact projection has an unsupported schema version")
    if artifact_data["problem_snapshot_schema_version"] != snapshot.schema_version:
        raise ValueError("AuthoringSnapshot artifact projection names a different snapshot schema")
    if not isinstance(artifact_data["payload"], dict):
        raise TypeError("AuthoringSnapshot artifact projection payload must be a mapping")
    expected_artifact = canonical_sha256(artifact_data)
    if artifact_hash != expected_artifact:
        raise ValueError(
            "AuthoringSnapshot.artifact_hash does not match its canonical artifact projection")
    return snapshot_hash


def prepare_compile_snapshot(
    problem: Any,
    time: Any,
    *,
    layout: Any,
    libraries: Any = (),
) -> AuthoringSnapshot:
    """Freeze authoring, then capture the complete effective compile transaction."""
    freeze = getattr(problem, "freeze", None)
    if not callable(freeze):
        raise TypeError("pops.compile requires a Case exposing freeze()")
    freeze()
    time_freeze = getattr(time, "freeze", None) if time is not None else None
    if callable(time_freeze):
        time_freeze()
    snapshot = build_authoring_snapshot(
        problem, layout=layout, time=time, libraries=libraries)
    validate_problem_snapshot(snapshot)
    return snapshot


def attach_problem_snapshot(compiled: Any, snapshot: Any) -> None:
    """Attach an already-keyed snapshot and seal the completed public artifact."""
    validate_problem_snapshot(snapshot)
    from pops.codegen.loader import CompiledModel

    is_model = isinstance(compiled, CompiledModel)
    existing = (
        object.__getattribute__(compiled, "__dict__").get("_problem_snapshot")
        if is_model else getattr(compiled, "_problem_snapshot", None)
    )
    if existing is not None and existing is not snapshot:
        raise ValueError("compiled artifact carries a different AuthoringSnapshot than its cache key")
    if existing is None:
        if is_model:
            object.__setattr__(compiled, "_problem_snapshot", snapshot)
        else:
            compiled._problem_snapshot = snapshot

    if is_model:
        # Do not dispatch through a potentially overridden/no-op subclass hook.
        CompiledModel._seal(compiled)
    elif hasattr(compiled, "_seal"):
        compiled._seal()


__all__ = [
    "AuthoringSnapshot", "build_problem_snapshot",
    "build_authoring_snapshot", "validate_problem_snapshot", "prepare_compile_snapshot",
    "attach_problem_snapshot", "SNAPSHOT_SCHEMA_VERSION", "ARTIFACT_SCHEMA_VERSION",
]
