"""Layer-neutral collective control plane for one shared restart checkpoint artifact.

The native Uniform/AMR codecs own field gathers, MPI transport and state mutation.  This module
only builds small deterministic control envelopes around one authenticated native communicator.
It never imports a Python MPI binding or executes a collective outside :mod:`pops._pops`.
"""
from __future__ import annotations

import os
from io import BytesIO
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pops._native_collectives import (
    allgather_value,
    broadcast_bytes,
    broadcast_value,
    encode_value,
    rank as native_rank,
    require_world,
    size as native_size,
)


@dataclass(frozen=True, slots=True)
class CheckpointTopology:
    """Exact rank topology carried by one installed RuntimeInstance."""

    rank: int
    size: int
    communicator: Any = None

    @property
    def distributed(self) -> bool:
        return self.communicator is not None


class InMemoryCheckpoint(Mapping[str, Any]):
    """Closed, object-free NPZ payload used by every restart rank.

    ``numpy.load`` is deliberately consumed while decoding the broadcast bytes.  Native engine
    adapters therefore never reopen the shared checkpoint path and never retain a lazy ``NpzFile``
    whose later access could perform rank-local filesystem I/O.
    """

    __slots__ = ("_arrays", "files")

    def __init__(self, arrays: Mapping[str, Any]) -> None:
        self._arrays = MappingProxyType(dict(arrays))
        self.files = tuple(self._arrays)

    def __getitem__(self, key: str) -> Any:
        return self._arrays[key]

    def __iter__(self):
        return iter(self._arrays)

    def __len__(self) -> int:
        return len(self._arrays)


def decode_checkpoint_bytes(payload: Any) -> InMemoryCheckpoint:
    """Decode exact NPZ bytes once into eager, immutable-by-contract arrays."""
    if not isinstance(payload, bytes) or not payload:
        raise TypeError("restart payload must be non-empty exact bytes")
    import numpy as np

    with np.load(BytesIO(payload), allow_pickle=False) as stored:
        names = tuple(stored.files)
        if len(names) != len(set(names)):
            raise ValueError("checkpoint NPZ contains duplicate array names")
        arrays = {}
        for name in names:
            if not isinstance(name, str) or not name:
                raise ValueError("checkpoint NPZ contains an invalid array name")
            value = np.asarray(stored[name])
            if value.dtype.hasobject:
                raise TypeError("checkpoint payload cannot contain object dtype")
            copied = np.array(value, copy=True, order="C")
            copied.setflags(write=False)
            arrays[name] = copied
    return InMemoryCheckpoint(arrays)


def checkpoint_topology(owner: Any) -> CheckpointTopology:
    """Project an installed ExecutionContext without inferring a communicator."""
    context = getattr(owner, "_execution_context", None)
    resource = getattr(context, "communicator", None)
    identity = getattr(resource, "identity", None)
    handle = getattr(resource, "handle", None)
    if identity == "serial":
        if handle is not None:
            raise ValueError("serial checkpoint context hides a communicator handle")
        return CheckpointTopology(0, 1)
    if identity is None:
        raise ValueError(
            "checkpoint requires the authenticated ExecutionContext installed by pops.bind"
        )
    if identity != "MPI_COMM_WORLD":
        raise ValueError(
            "distributed checkpoint requires the authenticated MPI_COMM_WORLD "
            "ExecutionContext resource"
        )
    native = require_world(handle)
    return CheckpointTopology(native_rank(native), native_size(native), native)


def canonical_checkpoint_path(value: Any, *, extension: str = ".npz") -> Path:
    """Return one lexical absolute path suitable for rank-to-rank equality checks."""
    if not isinstance(extension, str) or not extension.startswith(".") or "/" in extension:
        raise TypeError("checkpoint extension must be a canonical file suffix")
    text = os.fspath(value)
    if not isinstance(text, str) or not text or "\x00" in text:
        raise TypeError("checkpoint path must be non-empty filesystem text")
    path = Path(text)
    if path.suffix != extension:
        path = path.with_suffix(extension)
    return Path(os.path.abspath(os.path.normpath(os.fspath(path))))


def _error_record(error: BaseException) -> dict[str, str]:
    """Return one transport-safe error record without erasing its semantic family.

    MPI peers cannot re-raise another rank's exception object, but reducing every failure to a
    ``RuntimeError`` also destroys useful scientific contracts such as ``ValueError`` for an
    invalid checkpoint manifest.  Carry the exact qualified type for diagnostics and one closed
    builtin family for deterministic reconstruction on every rank.
    """
    if isinstance(error, FileNotFoundError):
        family = "FileNotFoundError"
    elif isinstance(error, PermissionError):
        family = "PermissionError"
    elif isinstance(error, OSError):
        family = "OSError"
    elif isinstance(error, TypeError):
        family = "TypeError"
    elif isinstance(error, ValueError):
        family = "ValueError"
    elif isinstance(error, KeyError):
        family = "KeyError"
    elif isinstance(error, IndexError):
        family = "IndexError"
    elif isinstance(error, NotImplementedError):
        family = "NotImplementedError"
    elif isinstance(error, AssertionError):
        family = "AssertionError"
    else:
        family = "RuntimeError"
    error_type = type(error)
    try:
        message = str(error)
    except BaseException:
        # Exception formatting is user-overridable. It must never fail before a peer enters the
        # matching broadcast/all-gather, otherwise one malformed rank-local error can deadlock the
        # remaining ranks. The exact qualified type still identifies the local cause.
        message = "<exception message unavailable>"
    return {
        "family": family,
        "type": "%s.%s" % (error_type.__module__, error_type.__qualname__),
        "message": message,
    }


_ERROR_FAMILIES: dict[str, type[BaseException]] = {
    "AssertionError": AssertionError,
    "FileNotFoundError": FileNotFoundError,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "NotImplementedError": NotImplementedError,
    "OSError": OSError,
    "PermissionError": PermissionError,
    "RuntimeError": RuntimeError,
    "TypeError": TypeError,
    "ValueError": ValueError,
}


def _validated_error_record(value: Any, *, phase: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"family", "type", "message"}:
        raise RuntimeError("checkpoint %s returned an invalid error record" % phase)
    if any(not isinstance(value[key], str) for key in ("family", "type", "message")):
        raise RuntimeError("checkpoint %s returned a non-text error record" % phase)
    if value["family"] not in _ERROR_FAMILIES or not value["type"]:
        raise RuntimeError("checkpoint %s returned an unknown error family" % phase)
    return value


def _raise_collective_failure(
    phase: str,
    failures: tuple[tuple[int, Mapping[str, str]], ...],
) -> None:
    """Raise one deterministic builtin family while retaining every rank-local cause."""
    if not failures:
        return
    families = {record["family"] for _rank, record in failures}
    error_type = _ERROR_FAMILIES[next(iter(families))] if len(families) == 1 else RuntimeError
    details = "; ".join(
        "rank %d: %s: %s" % (rank, record["type"], record["message"])
        for rank, record in failures
    )
    raise error_type("collective checkpoint %s failed: %s" % (phase, details))


def root_value(
    topology: CheckpointTopology,
    phase: str,
    producer: Callable[[], Any],
) -> Any:
    """Run one Python/filesystem decision on rank zero and broadcast its result or failure."""
    if not isinstance(phase, str) or not phase:
        raise TypeError("checkpoint phase must be non-empty text")
    if not callable(producer):
        raise TypeError("checkpoint root producer must be callable")
    envelope = None
    if topology.rank == 0:
        try:
            envelope = {"value": producer(), "error": None}
        except BaseException as error:
            if not topology.distributed:
                raise
            envelope = {"value": None, "error": _error_record(error)}
    if topology.distributed:
        envelope = broadcast_value(topology.communicator, envelope, root=0)
    else:
        # Keep serial and MPI semantics identical: generic control envelopes never carry bulk
        # bytes, even though no transport would otherwise force their encoding in serial.
        encode_value(envelope)
    if not isinstance(envelope, Mapping) or set(envelope) != {"value", "error"}:
        raise RuntimeError("checkpoint %s broadcast returned an invalid envelope" % phase)
    if envelope["error"] is not None:
        record = _validated_error_record(envelope["error"], phase=phase)
        _raise_collective_failure(phase, ((0, record),))
    return envelope["value"]


def root_effect(
    topology: CheckpointTopology,
    phase: str,
    operation: Callable[[], Any],
) -> Any:
    """Run rank-zero file I/O while broadcasting only completion/failure, never bulk data."""
    if not callable(operation):
        raise TypeError("checkpoint root operation must be callable")
    result = None
    failure = None
    if topology.rank == 0:
        try:
            result = operation()
        except BaseException as error:
            if not topology.distributed:
                raise
            failure = _error_record(error)
    if topology.distributed:
        failure = broadcast_value(topology.communicator, failure, root=0)
    if failure is not None:
        record = _validated_error_record(failure, phase=phase)
        _raise_collective_failure(phase, ((0, record),))
    return result


def root_bytes(
    topology: CheckpointTopology,
    phase: str,
    producer: Callable[[], bytes],
) -> bytes:
    """Read bytes on rank zero and broadcast them directly through the native C++ transport."""
    if not isinstance(phase, str) or not phase:
        raise TypeError("checkpoint phase must be non-empty text")
    if not callable(producer):
        raise TypeError("checkpoint root byte producer must be callable")
    payload = b""
    failure = None
    if topology.rank == 0:
        try:
            payload = producer()
            if not isinstance(payload, bytes) or not payload:
                raise TypeError("checkpoint root byte producer must return non-empty exact bytes")
        except BaseException as error:
            if not topology.distributed:
                raise
            failure = _error_record(error)
    if topology.distributed:
        failure = broadcast_value(topology.communicator, failure, root=0)
    if failure is not None:
        record = _validated_error_record(failure, phase=phase)
        _raise_collective_failure(phase, ((0, record),))
    if topology.distributed:
        payload = broadcast_bytes(topology.communicator, payload, root=0)
    return payload


def consensus(
    topology: CheckpointTopology,
    phase: str,
    *,
    error: BaseException | None = None,
    value: Any = None,
) -> tuple[Mapping[str, Any], ...]:
    """Convert every local phase result into one ordered all-rank decision.

    The next collective phase may start only after this function returns successfully.  In serial,
    the original exception type is preserved. In MPI, all ranks raise the same deterministic
    builtin semantic family and text; mixed families become ``RuntimeError`` while the message
    retains every rank-local exact exception type.
    """
    if error is not None and not isinstance(error, BaseException):
        raise TypeError("checkpoint consensus error must be an exception or None")
    if not topology.distributed:
        if error is not None:
            raise error
        return ({"rank": 0, "value": value, "error": None},)
    envelope = {
        "rank": topology.rank,
        "value": value,
        "error": None if error is None else _error_record(error),
    }
    rows = allgather_value(topology.communicator, envelope)
    if len(rows) != topology.size:
        raise RuntimeError(
            "checkpoint %s consensus returned %d ranks, expected %d"
            % (phase, len(rows), topology.size)
        )
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"rank", "value", "error"}:
            raise RuntimeError("checkpoint %s consensus row has an invalid schema" % phase)
        if row["rank"] != index:
            raise RuntimeError("checkpoint %s consensus rank order is invalid" % phase)
        normalized.append(row)
    failures = tuple(
        (
            int(row["rank"]),
            _validated_error_record(row["error"], phase=phase),
        )
        for row in normalized if row["error"] is not None
    )
    if failures:
        _raise_collective_failure(phase, failures)
    return tuple(normalized)


def collective_checkpoint_capture(
    owner: Any,
    phase_prefix: str,
    prepare: Callable[[], tuple[Any, str]],
    capture: Callable[[Any], tuple[Any, str]],
    publish: Callable[[Any], Any],
) -> Any:
    """Run a deadlock-safe checkpoint capture and rank-zero publication.

    ``prepare`` must be collective-free and return the opaque plan plus its content identity.
    Every rank agrees on that identity before ``capture`` may invoke its first native collective.
    ``capture`` returns an in-memory sealed artifact plus its restart identity; every rank agrees on
    that identity before ``publish`` performs any filesystem write on rank zero.
    """
    if not isinstance(phase_prefix, str) or not phase_prefix:
        raise TypeError("checkpoint capture phase prefix must be non-empty text")
    if not all(callable(callback) for callback in (prepare, capture, publish)):
        raise TypeError("checkpoint capture callbacks must be callable")
    topology = checkpoint_topology(owner)

    plan = None
    plan_identity = None
    prepare_error = None
    try:
        prepared = prepare()
        if type(prepared) is not tuple or len(prepared) != 2:
            raise TypeError("checkpoint prepare must return (plan, identity)")
        plan, plan_identity = prepared
        if not isinstance(plan_identity, str) or not plan_identity:
            raise TypeError("checkpoint capture-plan identity must be non-empty text")
    except BaseException as error:
        prepare_error = error
    rows = consensus(
        topology,
        "%s preflight" % phase_prefix,
        error=prepare_error,
        value=plan_identity,
    )
    if any(row["value"] != plan_identity for row in rows):
        raise RuntimeError(
            "collective checkpoint %s capture plans differ across ranks" % phase_prefix)

    artifact = None
    artifact_identity = None
    capture_error = None
    try:
        captured = capture(plan)
        if type(captured) is not tuple or len(captured) != 2:
            raise TypeError("checkpoint capture must return (artifact, identity)")
        artifact, artifact_identity = captured
        if not isinstance(artifact_identity, str) or not artifact_identity:
            raise TypeError("sealed checkpoint identity must be non-empty text")
    except BaseException as error:
        capture_error = error
    rows = consensus(
        topology,
        "%s sealed payload" % phase_prefix,
        error=capture_error,
        value=artifact_identity,
    )
    if any(row["value"] != artifact_identity for row in rows):
        raise RuntimeError(
            "collective checkpoint %s sealed payloads differ across ranks" % phase_prefix)

    return root_value(
        topology,
        "%s publication" % phase_prefix,
        lambda: publish(artifact),
    )


def _result_evidence(value: Any) -> Any:
    from pops.identity import Identity

    # Identity.to_data() is the canonical CBOR form and intentionally carries a raw 32-byte digest.
    # Structured MPI collectives deliberately refuse bytes so binary payloads cannot accidentally
    # bypass the direct byte transport.  Restart consensus needs only equality evidence, for which
    # the lossless, domain-qualified printable token is the exact representation.
    if type(value) is Identity:
        return value.token
    to_data = getattr(value, "to_data", None)
    return to_data() if callable(to_data) else value


def restore_checkpoint_payload(
    owner: Any,
    executor: Any,
    payload: bytes,
    *,
    phase_prefix: str = "native restart",
) -> Any:
    """Preflight and atomically apply one in-memory payload on the installed communicator.

    Every fallible preparation finishes with an all-rank consensus before the first native write.
    The accepted native snapshot remains rollback-capable through the apply and commit consensuses;
    only the final, non-fallible release discards it.
    """
    if not isinstance(phase_prefix, str) or not phase_prefix:
        raise TypeError("restart phase prefix must be non-empty text")
    topology = checkpoint_topology(owner)
    method_names = (
        "_prepare_checkpoint_restart",
        "_begin_checkpoint_restart",
        "_apply_checkpoint_restart",
        "_commit_checkpoint_restart",
        "_finalize_checkpoint_restart",
        "_rollback_checkpoint_restart",
    )
    methods: dict[str, Callable[..., Any]] = {}
    protocol_error = None
    try:
        missing = []
        for name in method_names:
            method = getattr(executor, name, None)
            if not callable(method):
                missing.append(name)
                continue
            methods[name] = method
        if missing:
            raise TypeError(
                "restart engine lacks the exact in-memory transaction protocol: %s"
                % ", ".join(missing)
            )
    except BaseException as error:
        protocol_error = error
    consensus(
        topology,
        "%s provider protocol" % phase_prefix,
        error=protocol_error,
        value="|".join(method_names) if protocol_error is None else None,
    )

    prepared = None
    prepare_error = None
    try:
        prepared = methods["_prepare_checkpoint_restart"](payload)
    except BaseException as error:
        prepare_error = error
    consensus(topology, "%s preflight" % phase_prefix, error=prepare_error)

    active = False
    begin_error = None
    try:
        methods["_begin_checkpoint_restart"]()
        active = True
    except BaseException as error:
        begin_error = error
    try:
        consensus(topology, "%s transaction begin" % phase_prefix, error=begin_error)
    except BaseException as original:
        rollback_error = None
        if active:
            try:
                methods["_rollback_checkpoint_restart"]()
            except BaseException as error:
                rollback_error = error
        try:
            consensus(
                topology,
                "%s begin rollback" % phase_prefix,
                error=rollback_error,
            )
        except BaseException as cleanup:
            add_note = getattr(original, "add_note", None)
            if callable(add_note):
                add_note("restart begin rollback also failed: %s" % cleanup)
        raise

    def rollback_after(original: BaseException, phase: str) -> None:
        rollback_error = None
        try:
            methods["_rollback_checkpoint_restart"]()
        except BaseException as error:
            rollback_error = error
        try:
            consensus(
                topology,
                "%s %s rollback" % (phase_prefix, phase),
                error=rollback_error,
            )
        except BaseException as cleanup:
            add_note = getattr(original, "add_note", None)
            if callable(add_note):
                add_note("restart rollback also failed: %s" % cleanup)

    result = None
    apply_error = None
    try:
        result = methods["_apply_checkpoint_restart"](prepared)
    except BaseException as error:
        apply_error = error
    try:
        rows = consensus(
            topology,
            "%s apply" % phase_prefix,
            error=apply_error,
            value=_result_evidence(result),
        )
        if any(row["value"] != rows[0]["value"] for row in rows[1:]):
            raise RuntimeError("%s ranks returned divergent restart evidence" % phase_prefix)
    except BaseException as original:
        rollback_after(original, "apply")
        raise

    commit_error = None
    try:
        methods["_commit_checkpoint_restart"]()
    except BaseException as error:
        commit_error = error
    try:
        consensus(topology, "%s commit" % phase_prefix, error=commit_error)
    except BaseException as original:
        rollback_after(original, "commit")
        raise

    # Finalization only releases snapshots that every rank has already agreed to commit.  Providers
    # must implement it as a no-throw release; the consensus turns a contract violation into one
    # coherent failure instead of allowing a peer to enter the next operation silently.
    finalize_error = None
    try:
        methods["_finalize_checkpoint_restart"]()
    except BaseException as error:
        finalize_error = error
    consensus(topology, "%s finalize" % phase_prefix, error=finalize_error)
    return result


def restore_checkpoint_path(
    owner: Any,
    executor: Any,
    path: Any,
    *,
    phase_prefix: str = "native restart",
) -> Any:
    """Collectively read and restore one shared checkpoint through native transports.

    This is the direct-engine counterpart of ``RestartV3``.  It deliberately carries no
    ``ConsumerGraph`` cursor state, but it uses the same rank-zero read, exact path agreement,
    byte broadcast and all-rank transactional restore as a bound ``RuntimeInstance``.
    """
    if not isinstance(phase_prefix, str) or not phase_prefix:
        raise TypeError("restart phase prefix must be non-empty text")
    topology = checkpoint_topology(owner)
    target = None
    target_text = None
    target_error = None
    try:
        target = canonical_checkpoint_path(path)
        target_text = str(target)
    except BaseException as error:
        target_error = error
    rows = consensus(
        topology,
        "%s target" % phase_prefix,
        error=target_error,
        value=target_text,
    )
    if target is None or target_text is None:
        raise RuntimeError("%s target consensus lost its local path" % phase_prefix)
    if any(row["value"] != target_text for row in rows):
        raise ValueError("%s target differs across ranks" % phase_prefix)
    payload = root_bytes(topology, "%s read" % phase_prefix, target.read_bytes)
    return restore_checkpoint_payload(
        owner, executor, payload, phase_prefix=phase_prefix)


__all__ = [
    "CheckpointTopology", "InMemoryCheckpoint",
    "canonical_checkpoint_path",
    "checkpoint_topology",
    "collective_checkpoint_capture",
    "consensus",
    "decode_checkpoint_bytes",
    "restore_checkpoint_path",
    "restore_checkpoint_payload",
    "root_effect",
    "root_bytes",
    "root_value",
]
