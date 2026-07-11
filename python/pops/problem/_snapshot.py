"""Frozen Problem snapshots and their reproducibility/compile identities (ADC-563).

``Problem.freeze()`` returns a :class:`ProblemSnapshot`: an inert, JSON-ready, array-free capture of
the whole assembly (blocks / fields / params / aux / outputs / constraints / time / layout) with a
stable ``.hash`` (sha256 over the canonical ``to_dict``). ``pops.compile`` freezes the Problem
before invoking the compiler. ``snapshot.hash`` remains the lossless reproducibility identity,
while ``snapshot.artifact_hash`` is the compile identity produced through explicit
``artifact_data()`` projections. Runtime defaults and report-only provenance remain visible in the
full snapshot without forcing a new binary.

Canonical projection is isolated in :mod:`pops.problem._snapshot_canonical`; this module owns only
the immutable value, hashing, construction, validation, and attachment lifecycle.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from pops.problem._snapshot_canonical import _canonical

#: Bumped when the full snapshot's canonical shape changes. The compile-only projection has its own
#: version below, so artifact-identity evolution does not rewrite the reproducibility schema.
SNAPSHOT_SCHEMA_VERSION = 5

#: Independent namespace for the compile-identity projection. Changing which declaration facts
#: affect generated artifacts bumps this version without rewriting the full snapshot schema.
ARTIFACT_SCHEMA_VERSION = 1


class ProblemSnapshot:
    """The frozen, JSON-ready capture of a :class:`~pops.problem.problem.Problem` (ADC-563).

    A plain inert value: :attr:`to_dict` is the canonical dict (deep, array-free, no runtime object)
    and :attr:`hash` is its stable sha256. Two snapshots of the same assembly have the same hash; a
    mutation before freeze changes it, a mutation after freeze is impossible (the Problem raises).
    ``pops.compile`` attaches it to the compiled handle (``compiled._problem_snapshot``) after the
    compile driver has included :attr:`artifact_hash` in the artifact hash, cache key, path and
    sidecar. :attr:`hash` remains the exact authored/reproducibility identity.
    """

    schema_version = SNAPSHOT_SCHEMA_VERSION
    __slots__ = ("_canonical_json", "_hash", "_artifact_canonical_json", "_artifact_hash")

    def __init__(
        self,
        payload: Any,
        *,
        handle_resolver: Any = None,
        artifact_payload: Any = None,
    ) -> None:
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

        # This is a separately versioned preimage, not a scrub of the full canonical dict. The
        # Problem builder supplies an explicit parameter/model projection; standalone snapshots use
        # the same raw payload and let objects opt in through artifact_data().
        artifact_source = payload if artifact_payload is None else artifact_payload
        canonical_artifact_payload = _canonical(
            artifact_source, handle_resolver=handle_resolver, artifact=True)
        if not isinstance(canonical_artifact_payload, dict):
            raise TypeError("ProblemSnapshot artifact payload must be a mapping")
        if "schema_version" in canonical_artifact_payload:
            raise ValueError(
                "ProblemSnapshot artifact payload cannot define reserved key 'schema_version'")
        artifact_envelope = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "problem_snapshot_schema_version": self.schema_version,
            "payload": canonical_artifact_payload,
        }
        artifact_json = json.dumps(
            artifact_envelope, sort_keys=True, separators=(",", ":"), allow_nan=False)
        object.__setattr__(self, "_artifact_canonical_json", artifact_json)
        object.__setattr__(
            self, "_artifact_hash",
            hashlib.sha256(artifact_json.encode("utf-8")).hexdigest())

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

    @property
    def artifact_hash(self) -> str:
        """Stable sha256 of compile-relevant data, excluding runtime bind values/defaults.

        This still covers kind, dtype, unit, domain, storage, phase/invalidation, constant values,
        ownership, and every non-parameter artifact input.
        """
        return self._artifact_hash

    def artifact_to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-ready copy of the versioned artifact projection."""
        return json.loads(self._artifact_canonical_json)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, ProblemSnapshot) and self.hash == other.hash

    def __hash__(self) -> int:
        return hash(self.hash)

    def __repr__(self) -> str:
        return "ProblemSnapshot(hash=%s..., artifact_hash=%s...)" % (
            self.hash[:12], self.artifact_hash[:12])


def build_problem_snapshot(problem: Any) -> Any:
    """Build the :class:`ProblemSnapshot` of @p problem (the frozen input to the compile cache key).

    Reads the Problem's raw typed registries (not its intentionally concise inspection view) and
    canonicalises them into a JSON-ready, deep, inert payload. It computes nothing on a grid and
    imports no ``_pops``. ``.hash`` preserves the exact authored capture and ``.artifact_hash`` is
    the driver-owned compile identity, so a runtime default can change without recompiling while a
    constant or ABI-relevant declaration change cannot silently reuse an artifact."""
    from pops.problem._snapshot_payload import (
        problem_snapshot_artifact_payload,
        problem_snapshot_payload,
    )

    payload = problem_snapshot_payload(problem)
    artifact_payload = problem_snapshot_artifact_payload(problem)
    return ProblemSnapshot(
        payload,
        handle_resolver=problem.resolve,
        artifact_payload=artifact_payload,
    )


def validate_problem_snapshot(snapshot: Any) -> str:
    """Validate both identities and return the full reproducibility hash.

    The compile driver reads :attr:`ProblemSnapshot.artifact_hash` only after this function has
    authenticated the exact full snapshot and separately versioned artifact projection.
    """
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
    artifact_hash = snapshot.artifact_hash
    if not isinstance(artifact_hash, str) or len(artifact_hash) != 64 \
            or any(char not in "0123456789abcdef" for char in artifact_hash):
        raise ValueError(
            "ProblemSnapshot.artifact_hash must be exactly 64 lowercase hexadecimal characters")
    artifact_data = snapshot.artifact_to_dict()
    if not isinstance(artifact_data, dict) or set(artifact_data) != {
            "artifact_schema_version", "problem_snapshot_schema_version", "payload"}:
        raise ValueError("ProblemSnapshot artifact projection has an invalid envelope")
    if artifact_data["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("ProblemSnapshot artifact projection has an unsupported schema version")
    if artifact_data["problem_snapshot_schema_version"] != snapshot.schema_version:
        raise ValueError("ProblemSnapshot artifact projection names a different snapshot schema")
    if not isinstance(artifact_data["payload"], dict):
        raise TypeError("ProblemSnapshot artifact projection payload must be a mapping")
    artifact_json = json.dumps(
        artifact_data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    expected_artifact = hashlib.sha256(artifact_json.encode("utf-8")).hexdigest()
    if artifact_hash != expected_artifact:
        raise ValueError(
            "ProblemSnapshot.artifact_hash does not match its canonical artifact projection")
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
    "ARTIFACT_SCHEMA_VERSION",
]
