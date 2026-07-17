"""Backend-independent scientific-output identity and publication transaction."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pops.identity import Identity, make_identity

from pops.output.data import OutputRequest, OutputSnapshot, array_evidence
from pops._native_collectives import (
    barrier as native_barrier,
    broadcast_value,
    rank as native_rank,
    require_world,
    size as native_size,
)


OUTPUT_SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def writer_execution_capability(
    execution_context: Any,
    mode: Any,
    *,
    provider_id: str,
) -> dict[str, Any]:
    """Authenticate one writer's topology without consulting process-global MPI state."""
    from pops._platform_contracts import ExecutionContext
    from pops.output._consumer_contracts import ParallelMode

    if type(execution_context) is not ExecutionContext:
        raise TypeError("writer preflight requires an exact ExecutionContext")
    if type(mode) is not ParallelMode:
        raise TypeError("writer preflight requires an exact ParallelMode")
    if not isinstance(provider_id, str) or not provider_id or provider_id.strip() != provider_id:
        raise TypeError("writer preflight provider_id must be canonical text")
    communicator = execution_context.communicator
    serial = communicator.identity == "serial"
    handle = communicator.handle
    if serial:
        if handle is not None:
            raise ValueError("serial writer preflight rejects a hidden communicator handle")
        size = 1
    else:
        native = require_world(handle)
        size = native_size(native)
    if mode is ParallelMode.SERIAL and not serial:
        raise ValueError("SERIAL scientific output requires a serial ExecutionContext")
    if mode is not ParallelMode.SERIAL and serial:
        raise ValueError(
            "%s scientific output requires a distributed ExecutionContext" % mode.name)
    return {
        "schema_version": 1,
        "provider_id": provider_id,
        "parallel_mode": mode.value,
        "communicator": communicator.identity,
        "size": size,
    }


def json_text(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def identity_from_token(token: Any, domain: str, where: str) -> Identity:
    try:
        result = Identity.from_token(token)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s has an invalid identity" % where) from exc
    if result.domain != domain:
        raise ValueError("%s must use the %r identity domain" % (where, domain))
    return result


def deterministic_target(
    directory: Any,
    prefix: Any,
    request: OutputRequest,
    snapshot: OutputSnapshot,
    extension: str,
) -> Path:
    """Return the sole deterministic, filesystem-bounded output filename.

    Human-readable prefixes are deliberately bounded, while the digest covers every full
    identity-bearing input.  Long consumer or clock identities therefore cannot exceed common
    ``NAME_MAX`` limits and cannot collide merely because their readable prefixes are equal.
    """
    root = Path(directory)
    clean_prefix = _SAFE_NAME.sub("-", str(prefix)).strip("-")
    clean_consumer = _SAFE_NAME.sub("-", request.consumer_id).strip("-")
    clean_clock = _SAFE_NAME.sub("-", snapshot.clock.clock_id).strip("-")
    if not clean_prefix or not clean_consumer or not clean_clock:
        raise ValueError("output filename parts must contain a safe non-empty token")
    if (not extension.startswith(".") or "/" in extension or "\\" in extension
            or len(extension.encode("utf-8")) > 32):
        raise ValueError("output extension must be a simple suffix")
    target_identity = make_identity("scientific-output-target", {
        "prefix": str(prefix),
        "consumer_id": request.consumer_id,
        "clock": snapshot.clock.to_data(),
        "publication_selection": request.publication_data(),
        "extension": extension,
    })
    from pops.output._consumer_contracts import ParallelMode

    rank_part = (
        "__r%06d" % request.rank
        if request.parallel_mode is ParallelMode.PER_RANK else ""
    )
    name = "%s__%s__s%09d%s__%s%s" % (
        clean_prefix[:40],
        clean_consumer[:40],
        snapshot.clock.macro_step,
        rank_part,
        target_identity.hexdigest,
        extension,
    )
    return root / name


def manifest(
    format_name: str,
    snapshot: OutputSnapshot,
    request: OutputRequest,
    arrays: dict[str, Any],
    *,
    snapshot_data: dict[str, Any] | None = None,
    datasets: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Identity]:
    base = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "format": format_name,
        "snapshot": snapshot_data if snapshot_data is not None else snapshot.to_data(request),
        "datasets": datasets or {},
        "arrays": {name: arrays[name] for name in sorted(arrays)},
    }
    identity = make_identity("scientific-output", base)
    return dict(base, output_identity=identity.token), identity


def authenticate_manifest(
    value: Any,
    format_name: str,
) -> tuple[dict[str, Any], Identity]:
    if not isinstance(value, dict):
        raise TypeError("scientific output manifest must be a mapping")
    required = {
        "schema_version", "format", "snapshot", "datasets", "arrays", "output_identity",
    }
    if set(value) != required:
        raise ValueError("scientific output manifest keys are not exact")
    if value["schema_version"] != OUTPUT_SCHEMA_VERSION or value["format"] != format_name:
        raise ValueError("scientific output schema/format mismatch")
    supplied = identity_from_token(
        value["output_identity"], "scientific-output", "output_identity")
    base = {key: value[key] for key in required - {"output_identity"}}
    expected = make_identity("scientific-output", base)
    if supplied != expected:
        raise ValueError("scientific output manifest identity mismatch")
    return value, expected


@dataclass(frozen=True, slots=True)
class OutputPublicationReceipt:
    path: Path
    format: str
    output_identity: Identity
    selection_identity: Identity


class WriterSession(Protocol):
    """Public structural transaction returned by a scientific-output writer.

    Implementations need not inherit from this protocol.  Construction and ``authority`` access
    must be effect-free; the runtime authenticates every rank before calling ``stage``.
    """

    @property
    def authority(self) -> dict[str, Any]: ...

    @property
    def identity(self) -> Identity: ...

    def stage(self) -> None: ...

    def abort_prepare(self) -> None: ...

    def publish(self) -> OutputPublicationReceipt | None: ...

    def rollback(self) -> None: ...


class ScientificWriter(Protocol):
    """Public structural writer protocol implemented by format providers."""

    def preflight(self, execution_context: Any) -> dict[str, Any]: ...

    def prepare_session(
        self,
        snapshot: OutputSnapshot,
        request: OutputRequest,
        target: Any,
        *,
        communicator: Any = None,
    ) -> WriterSession: ...


_SESSION_AUTHORITY_KEYS = frozenset({
    "schema_version", "session_identity", "format", "parallel_mode", "rank", "size",
    "role", "target", "selection_identity",
})


def writer_session_authority(
    format_name: str,
    request: OutputRequest,
    target: Any,
) -> dict[str, Any]:
    """Build canonical effect-free authority for one rank's writer session."""
    from pops.output._consumer_contracts import ParallelMode

    if not isinstance(format_name, str) or not format_name \
            or format_name.strip() != format_name:
        raise TypeError("writer session format must be canonical text")
    if type(request) is not OutputRequest:
        raise TypeError("writer session authority requires an exact OutputRequest")
    path = Path(target).expanduser().resolve()
    mode = request.parallel_mode
    role = {
        ParallelMode.SERIAL: "serial",
        ParallelMode.ROOT: "root" if request.rank == 0 else "participant",
        ParallelMode.COLLECTIVE: "collective",
        ParallelMode.PER_RANK: "local",
    }[mode]
    base = {
        "schema_version": 1,
        "format": format_name,
        "parallel_mode": mode.value,
        "rank": request.rank,
        "size": request.size,
        "role": role,
        "target": path.as_posix(),
        "selection_identity": request.publication_identity.token,
    }
    identity = make_identity("scientific-output-writer-session", base)
    return dict(base, session_identity=identity.token)


def authenticate_writer_session(session: Any) -> dict[str, Any]:
    """Validate a writer session structurally without naming a concrete backend type."""
    required_methods = ("stage", "abort_prepare", "publish", "rollback")
    if any(not callable(getattr(session, name, None)) for name in required_methods):
        raise TypeError(
            "scientific output session must implement stage(), abort_prepare(), "
            "publish(), and rollback()")
    first = getattr(session, "authority", None)
    second = getattr(session, "authority", None)
    if type(first) is not dict or type(second) is not dict or first != second:
        raise TypeError("writer session authority must be one deterministic dict")
    if set(first) != _SESSION_AUTHORITY_KEYS:
        raise ValueError("writer session authority keys are not exact")
    if first["schema_version"] != 1:
        raise ValueError("writer session authority schema_version must be 1")
    if first["parallel_mode"] not in {"serial", "root", "collective", "per_rank"}:
        raise ValueError("writer session authority has an invalid parallel mode")
    rank, size = first["rank"], first["size"]
    if type(rank) is not int or type(size) is not int or size < 1 or rank not in range(size):
        raise ValueError("writer session authority has an invalid rank topology")
    expected_role = {
        "serial": "serial",
        "root": "root" if rank == 0 else "participant",
        "collective": "collective",
        "per_rank": "local",
    }[first["parallel_mode"]]
    if first["role"] != expected_role:
        raise ValueError("writer session role differs from its parallel mode and rank")
    for key in ("format", "target", "selection_identity"):
        value = first[key]
        if not isinstance(value, str) or not value or value.strip() != value:
            raise TypeError("writer session %s must be canonical text" % key)
    supplied = identity_from_token(
        first["session_identity"],
        "scientific-output-writer-session",
        "writer session identity",
    )
    base = {key: first[key] for key in _SESSION_AUTHORITY_KEYS - {"session_identity"}}
    expected = make_identity("scientific-output-writer-session", base)
    if supplied != expected:
        raise ValueError("writer session identity does not authenticate its authority")
    if getattr(session, "identity", None) != expected:
        raise ValueError("writer session identity property differs from its authority")
    return first


class _StagedOutputFile:
    """Verified temporary scientific file, not yet attached to a consumer effect."""

    __slots__ = (
        "temporary", "target", "format", "output_identity", "selection_identity",
        "_verify", "_published", "_discarded", "_created_target", "_target_owner",
        "_temporary_owner", "_communicator",
    )

    def __init__(
        self,
        temporary: Any,
        target: Any,
        *,
        format: str,
        output_identity: Identity,
        selection_identity: Identity,
        verify: Callable[[Any], Any],
        communicator: Any = None,
        temporary_owner: tuple[int, int] | None = None,
    ) -> None:
        self.temporary, self.target = Path(temporary), Path(target)
        self.format = format
        self.output_identity, self.selection_identity = output_identity, selection_identity
        self._verify, self._communicator = verify, communicator
        self._published = self._discarded = False
        self._created_target = False
        self._target_owner: tuple[int, int] | None = None
        if communicator is None:
            if temporary_owner is not None:
                raise ValueError(
                    "serial scientific output rejects a detached temporary inode authority")
            temporary_stat = self.temporary.lstat()
            temporary_owner = (
                int(temporary_stat.st_dev),
                int(temporary_stat.st_ino),
            )
        elif (
            type(temporary_owner) is not tuple
            or len(temporary_owner) != 2
            or any(type(item) is not int or item < 0 for item in temporary_owner)
        ):
            raise ValueError(
                "collective scientific output requires the rank-zero temporary inode authority")
        self._temporary_owner = temporary_owner

    def _rank(self) -> int:
        return 0 if self._communicator is None else native_rank(self._communicator)

    def _barrier(self) -> None:
        if self._communicator is not None:
            native_barrier(self._communicator)

    def _unlink_temporary_owned(self) -> None:
        try:
            current = self.temporary.lstat()
        except FileNotFoundError:
            return
        owner = (int(current.st_dev), int(current.st_ino))
        if owner != self._temporary_owner:
            raise RuntimeError(
                "scientific output refuses to delete a replaced temporary at %s"
                % self.temporary)
        self.temporary.unlink()

    def publish(self) -> OutputPublicationReceipt:
        if self._discarded:
            raise RuntimeError("discarded output cannot be published")
        if self._published:
            return OutputPublicationReceipt(
                self.target, self.format, self.output_identity, self.selection_identity)
        self._barrier()
        failure = None
        if self._rank() == 0:
            try:
                self._verify(self.temporary)
                self.target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    staged = self.temporary.lstat()
                    owner = (int(staged.st_dev), int(staged.st_ino))
                    if owner != self._temporary_owner:
                        raise RuntimeError(
                            "scientific output staging inode changed before publication")
                    os.link(self.temporary, self.target)
                    self._created_target = True
                    self._target_owner = owner
                except FileExistsError:
                    if hashlib.sha256(self.temporary.read_bytes()).digest() != hashlib.sha256(
                            self.target.read_bytes()).digest():
                        raise FileExistsError(
                            "scientific output collision at deterministic target %s" % self.target
                        ) from None
                self._unlink_temporary_owned()
            except Exception as exc:
                failure = "%s: %s" % (type(exc).__name__, exc)
        if self._communicator is not None:
            failure = broadcast_value(self._communicator, failure, root=0)
        if failure is not None:
            if self._communicator is None and failure.startswith("FileExistsError:"):
                raise FileExistsError(failure.split(": ", 1)[1])
            raise RuntimeError("collective output publication failed: %s" % failure)
        self._barrier()
        self._published = True
        return OutputPublicationReceipt(
            self.target, self.format, self.output_identity, self.selection_identity)

    def discard(self) -> None:
        if self._published or self._discarded:
            return
        self._barrier()
        failure = None
        if self._rank() == 0:
            try:
                self._unlink_temporary_owned()
            except Exception as exc:
                failure = "%s: %s" % (type(exc).__name__, exc)
        if self._communicator is not None:
            failure = broadcast_value(self._communicator, failure, root=0)
        self._barrier()
        if failure is not None:
            raise RuntimeError("collective output discard failed: %s" % failure)
        self._discarded = True

    def rollback(self) -> None:
        """Compensate a staged or published output without deleting a pre-existing artifact."""
        if self._discarded:
            return
        self._barrier()
        failure = None
        if self._rank() == 0:
            try:
                self._unlink_temporary_owned()
                if self._created_target:
                    try:
                        current = self.target.lstat()
                    except FileNotFoundError:
                        current = None
                    if current is not None:
                        owner = (int(current.st_dev), int(current.st_ino))
                        if owner != self._target_owner:
                            raise RuntimeError(
                                "scientific output rollback refused a replaced target at %s"
                                % self.target)
                        self.target.unlink()
            except Exception as exc:
                failure = "%s: %s" % (type(exc).__name__, exc)
        if self._communicator is not None:
            failure = broadcast_value(self._communicator, failure, root=0)
        self._barrier()
        if failure is not None:
            raise RuntimeError("collective output rollback failed: %s" % failure)
        self._published = False
        self._discarded = True


class OutputWriterSession:
    """Built-in file-session implementation; custom writers may remain fully structural."""

    __slots__ = ("_authority", "_identity", "_stage_file", "_staged", "_aborted")

    def __init__(
        self,
        authority: dict[str, Any],
        stage_file: Callable[[], _StagedOutputFile] | None,
    ) -> None:
        self._authority = dict(authority)
        self._identity = identity_from_token(
            self._authority.get("session_identity"),
            "scientific-output-writer-session",
            "writer session identity",
        )
        self._stage_file = stage_file
        self._staged: _StagedOutputFile | None = None
        self._aborted = False
        authenticate_writer_session(self)

    @property
    def authority(self) -> dict[str, Any]:
        return dict(self._authority)

    @property
    def identity(self) -> Identity:
        return self._identity

    @property
    def temporary(self) -> Path | None:
        return None if self._staged is None else self._staged.temporary

    @property
    def target(self) -> Path:
        return Path(self._authority["target"])

    def stage(self) -> None:
        if self._aborted:
            raise RuntimeError("aborted writer session cannot be staged")
        if self._staged is not None or self._stage_file is None:
            return
        staged = self._stage_file()
        if type(staged) is not _StagedOutputFile:
            raise TypeError("built-in writer stage must return its private staged file")
        self._staged = staged

    def abort_prepare(self) -> None:
        if self._aborted:
            return
        if self._staged is not None:
            self._staged.discard()
        self._aborted = True

    def publish(self) -> OutputPublicationReceipt | None:
        if self._stage_file is None:
            return None
        if self._staged is None:
            raise RuntimeError("writer session must be staged before publication")
        return self._staged.publish()

    def rollback(self) -> None:
        if self._staged is not None:
            self._staged.rollback()
        self._aborted = True


@dataclass(frozen=True, slots=True)
class ReopenedOutput:
    manifest: dict[str, Any]
    arrays: dict[str, Any]
    output_identity: Identity

    def require_selection(self, request: OutputRequest) -> ReopenedOutput:
        recorded = self.manifest["snapshot"]["selection"]
        if recorded != request.publication_data():
            raise ValueError("reopened output selection differs from the requested selection")
        return self


def temporary_path(target: Path) -> Path:
    """Create one local staging file; distributed ownership is handled by its writer."""
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=".%s." % target.name,
        suffix=".prepared",
        dir=str(target.parent),
    )
    os.close(descriptor)
    return Path(name)


def piece_payload(
    snapshot: OutputSnapshot,
    request: OutputRequest,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Encode exact pieces without padding sparse AMR cells into physical values.

    Each piece remains a distinct identity-bearing array and its half-open global bounds are
    retained in the dataset map.  The same representation therefore works for complete serial/root
    snapshots and sparse collective/per-rank snapshots without a second storage convention.
    """
    arrays: dict[str, Any] = {}
    datasets: dict[str, Any] = {"fields": {}, "geometries": {}}
    fields = snapshot.select(request)
    for field_index, field in enumerate(fields):
        pieces = []
        for piece_index, piece in enumerate(field.pieces):
            name = "field_%04d_piece_%04d" % (field_index, piece_index)
            arrays[name] = piece.values
            pieces.append({
                "name": name,
                "lower": list(piece.lower),
                "upper": list(piece.upper),
                "global_box_index": piece.global_box_index,
                "owner_rank": piece.owner_rank,
                "replicated": piece.replicated,
            })
        datasets["fields"][field.key.identity.token] = {
            "global_shape": list(field.global_shape),
            "dtype": field.array_dtype,
            "pieces": pieces,
        }
    geometries = selected_geometries(snapshot, request, fields)
    for index, geometry in enumerate(sorted(geometries.values(), key=lambda item: item.key)):
        coverage = "geometry_%04d_coverage" % index
        valid = "geometry_%04d_valid" % index
        volumes = "geometry_%04d_volumes" % index
        arrays[coverage], arrays[valid], arrays[volumes] = (
            geometry.coverage, geometry.valid_cells, geometry.cell_volumes)
        datasets["geometries"]["%s#%d" % geometry.key] = {
            "coverage": coverage,
            "valid_cells": valid,
            "cell_volumes": volumes,
        }
    evidence = {name: array_evidence(value) for name, value in arrays.items()}
    return arrays, datasets, evidence


def field_values_on_mask(field: Any, mask: Any, *, require_piece_subset: bool) -> Any:
    """Return values in row-major mask order from exact sparse pieces.

    No value is synthesized outside the mask.  This is the common sparse projection used by VTK
    and composite diagnostics; missing or duplicate ownership fails before publication.
    """
    import numpy as np

    selected = np.asarray(mask, dtype=np.bool_)
    if selected.shape != field.global_shape:
        raise ValueError("field selection mask differs from global_shape")
    ordinals = np.full(field.global_shape, -1, dtype=np.int64)
    ordinals[selected] = np.arange(int(np.count_nonzero(selected)), dtype=np.int64)
    components = len(field.component_names)
    shape = (int(np.count_nonzero(selected)), components) if components else \
        (int(np.count_nonzero(selected)),)
    result = np.empty(shape, dtype=np.dtype(field.array_dtype))
    written = np.zeros(shape[0], dtype=np.bool_)
    for piece in field.pieces:
        jlo, ilo = piece.lower
        jhi, ihi = piece.upper
        local_mask = selected[jlo:jhi, ilo:ihi]
        if require_piece_subset and not np.all(local_mask):
            raise ValueError("field piece extends outside its exact valid geometry mask")
        target = ordinals[jlo:jhi, ilo:ihi][local_mask]
        if np.any(written[target]):
            raise ValueError("field pieces overlap on the selected geometry mask")
        if components:
            result[target, :] = piece.values[:, local_mask].T
        else:
            result[target] = piece.values[local_mask]
        written[target] = True
    if not np.all(written):
        raise ValueError("field pieces do not cover the exact selected geometry mask")
    result.setflags(write=False)
    return result


def validate_field_pieces(
    field: Any,
    geometry: Any,
    *,
    complete: bool,
    rank: int | None = None,
    size: int | None = None,
) -> None:
    """Prove that pieces are a non-overlapping subset/partition of valid geometry boxes."""
    pieces = sorted(field.pieces, key=lambda piece: (piece.lower, piece.upper))
    boxes = tuple(geometry.boxes)
    active: list[Any] = []
    covered = 0
    for piece in pieces:
        jlo, ilo = piece.lower
        jhi, ihi = piece.upper
        if piece.global_box_index >= len(boxes):
            raise ValueError("field global_box_index lies outside exact geometry boxes")
        if (jlo, ilo, jhi, ihi) != boxes[piece.global_box_index]:
            raise ValueError("field piece differs from its indexed exact geometry box")
        if rank is not None and piece.owner_rank != rank:
            raise ValueError("field piece owner_rank differs from its publication rank")
        if size is not None and piece.owner_rank >= size:
            raise ValueError("field piece owner_rank lies outside publication topology")
        if complete and piece.replicated and piece.owner_rank != 0:
            raise ValueError("complete field uses a non-root replicated piece authority")
        active = [other for other in active if other.upper[0] > jlo]
        if any(not (ihi <= other.lower[1] or other.upper[1] <= ilo) for other in active):
            raise ValueError("field pieces overlap")
        active.append(piece)
        covered += (jhi - jlo) * (ihi - ilo)
    if complete:
        expected = sum(
            (jhi - jlo) * (ihi - ilo)
            for jlo, ilo, jhi, ihi in boxes
        )
        if covered != expected:
            raise ValueError("field pieces do not exactly cover the valid geometry boxes")
        if {piece.global_box_index for piece in pieces} != set(range(len(boxes))):
            raise ValueError("field pieces do not authenticate every exact geometry box")


def selected_geometries(
    snapshot: OutputSnapshot,
    request: OutputRequest,
    fields: Any,
) -> dict[Any, Any]:
    geometries = {
        snapshot.geometry(field.key).key: snapshot.geometry(field.key)
        for field in fields
    }
    diagnostic_layouts = {item.layout_identity.token for item in request.diagnostics}
    geometries.update({
        item.key: item
        for item in snapshot.geometries
        if item.layout_identity.token in diagnostic_layouts
    })
    return geometries


__all__ = [
    "OUTPUT_SCHEMA_VERSION", "OutputPublicationReceipt", "OutputWriterSession",
    "ScientificWriter", "WriterSession", "ReopenedOutput",
    "authenticate_writer_session",
    "deterministic_target", "field_values_on_mask", "piece_payload",
    "validate_field_pieces", "writer_execution_capability", "writer_session_authority",
]
