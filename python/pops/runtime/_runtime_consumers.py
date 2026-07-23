"""Runtime-owned ConsumerGraph publication against accepted native state."""
from __future__ import annotations

import os
import math
import json
import stat
import tempfile
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from pops._native_collectives import (
    allgather_value,
    rank as native_rank,
    require_world,
    size as native_size,
)
from pops._frozen_data import thaw_data
from pops.identity import Identity, make_identity
from pops.mesh._layout_plan_contracts import (
    CARTESIAN_CELL_AREA,
    POLAR_ANNULUS_CELL_AREA,
    NormalizedGeometry,
)
from pops.output.data import (
    _NATIVE_GEOMETRY_ARRAYS,
    _NativeCompositeIntegral,
    _composite_integral_authority_identity,
    _field_family_identity,
    ArrayPiece,
    DiagnosticKey,
    DiagnosticPayload,
    FieldKey,
    FieldPayload,
    LevelGeometry,
    OutputClock,
    OutputProvenance,
    OutputRequest,
    OutputSnapshot,
)
from pops.output.observers import (
    ObserverFrame,
    ObserverRun,
)
from pops.output._consumer_contracts import ConsumerKind, ParallelMode
from pops.output._writers.common import (
    _OutputRecoveryRequired,
    _StagedOutputFile,
    _StagingAuthority,
    _exception_text,
    deterministic_target,
)

from ._consumer import (
    AcceptedSideEffect,
    ConsumerPublisher,
    PreparedPublication,
    PublicationReceipt,
)
from ._component_execution_context import component_execution_data
from ._output_publisher import ConsumerOutputPublisher, OutputPreparation
from ._observer_runtime import (
    _DetachedObserverFrame,
    _authenticated_detached_frame,
    _detach_owned_observer_frame,
    ObserverDeliveryReport,
    PostCommitObserverQueue,
    PostCommitObserverWorker,
)


_BUILTIN_CATALYST_PROCESS_LOCK = threading.Lock()
_BUILTIN_CATALYST_PROCESS_STARTED = False


def _reserve_builtin_catalyst_process_lifecycle() -> None:
    """Reserve Catalyst's process-global initialize/finalize lifecycle exactly once."""

    global _BUILTIN_CATALYST_PROCESS_STARTED
    with _BUILTIN_CATALYST_PROCESS_LOCK:
        if _BUILTIN_CATALYST_PROCESS_STARTED:
            raise RuntimeError(
                "the built-in Catalyst lifecycle has already started in this OS process; "
                "launch a new process for another Catalyst simulation run")
        _BUILTIN_CATALYST_PROCESS_STARTED = True


def _block_name(reference: Any, names: tuple[str, ...]) -> str:
    block = getattr(reference, "block_ref", None)
    local_id = getattr(block, "local_id", None)
    if local_id in names:
        return local_id
    if len(names) == 1:
        return names[0]
    raise ValueError("consumer reference has no exact installed block owner")


def _conservative_metadata(owner: Any, block: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read exact component names and physical roles from the compiled artifact authority."""
    from pops.codegen._artifact_models import artifact_model_metadata

    rows = [
        row for row in artifact_model_metadata(owner._install_plan.artifact)
        if row.block_name == block
    ]
    if len(rows) != 1 or not rows[0].cons_names \
            or len(rows[0].cons_roles) != len(rows[0].cons_names):
        raise ValueError("installed block %r has no exact conservative component order" % block)
    return rows[0].cons_names, rows[0].cons_roles


def _conservative_names(owner: Any, block: str) -> tuple[str, ...]:
    return _conservative_metadata(owner, block)[0]


def _diagnostic_record_name(payload: DiagnosticPayload) -> str:
    """Exact inspection key; distinct level/role declarations must never overwrite each other."""
    return "%s:%s:%s" % (
        payload.key.reference.qualified_id,
        payload.key.reduction,
        payload.key.state_id,
    )


def _identity_payload(value: Any, *, path: str = "layout") -> Any:
    """Project strict layout JSON into the float-free identity value language."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("%s contains a non-finite binary64 value" % path)
        return {"binary64": value.hex()}
    if isinstance(value, Mapping):
        return {
            key: _identity_payload(item, path="%s.%s" % (path, key))
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [
            _identity_payload(item, path="%s[%d]" % (path, index))
            for index, item in enumerate(value)
        ]
    return value


def _layout_identity(layout: Any) -> Identity:
    return make_identity("layout", _identity_payload(layout.to_data()))


_NATIVE_CELL_MEASURES = frozenset({
    CARTESIAN_CELL_AREA,
    POLAR_ANNULUS_CELL_AREA,
})


def _target(uri: str, format_data: Mapping[str, Any], format_name: str,
            snapshot: OutputSnapshot,
            request: OutputRequest,
            consumer_name: str, output_root: Any) -> Path:
    path = Path(uri)
    if output_root is not None:
        path = Path(output_root) / path
    return deterministic_target(
        path,
        consumer_name,
        request,
        snapshot,
        format_data["extension"],
        format_data=format_data,
        format_name=format_name,
    )


def _execution_topology(owner: Any) -> tuple[int, int, Any]:
    """Return the exact installed rank topology without consulting process globals."""
    communicator = owner._execution_context.communicator
    if communicator.identity == "serial":
        if communicator.handle is not None:
            raise ValueError("serial ExecutionContext hides a communicator handle")
        return 0, 1, None
    handle = communicator.handle
    native = require_world(handle)
    return native_rank(native), native_size(native), native


def _post_commit_root_consensus(
    communicator: Any,
    *,
    rank: int,
    size: int,
    error: str | None,
    phase: str,
) -> None:
    """Reach exactly one ROOT status collective before exposing any local failure."""

    rows = allgather_value(communicator, {"rank": rank, "error": error})
    if len(rows) != size or any(
            not isinstance(row, Mapping)
            or set(row) != {"rank", "error"}
            or row["rank"] != owner_rank
            or (row["error"] is not None and not isinstance(row["error"], str))
            for owner_rank, row in enumerate(rows)):
        raise RuntimeError("ROOT post-commit %s returned a malformed envelope" % phase)
    failures = [
        "rank %d: %s" % (owner_rank, row["error"])
        for owner_rank, row in enumerate(rows)
        if row["error"] is not None
    ]
    if failures:
        raise RuntimeError(
            "ROOT post-commit %s failed: %s" % (phase, "; ".join(failures)))


class _PreparedDiagnostic(PreparedPublication):
    def __init__(self, effect: AcceptedSideEffect, values: tuple[DiagnosticPayload, ...],
                 publish: Callable[
                     [AcceptedSideEffect, tuple[DiagnosticPayload, ...]], None],
                 discard: Callable[[AcceptedSideEffect], None],
                 rollback: Callable[
                     [AcceptedSideEffect, tuple[DiagnosticPayload, ...]], None]) -> None:
        self._effect, self._values = effect, values
        self._publish, self._discard, self._rollback = publish, discard, rollback
        self._published = self._discarded = False

    @property
    def effect_identity(self) -> Identity:
        return self._effect.identity

    @property
    def payload_identity(self) -> Identity:
        return self._effect.payload.identity

    def publish(self) -> PublicationReceipt:
        if self._discarded:
            raise RuntimeError("discarded diagnostic cannot be published")
        if not self._published:
            self._published = True
            try:
                self._publish(self._effect, self._values)
            except BaseException as error:
                try:
                    self._rollback(self._effect, self._values)
                except BaseException as cleanup_error:
                    add_note = getattr(error, "add_note", None)
                    if callable(add_note):
                        add_note("diagnostic publication rollback also failed: %s" % cleanup_error)
                self._published = False
                self._discarded = True
                raise
        artifact = make_identity("runtime-diagnostic-publication", {
            "effect": self.effect_identity.token,
            "values": [value.to_data() for value in self._values],
        })
        return PublicationReceipt(
            self.effect_identity, self.payload_identity,
            "pops.runtime-diagnostic.v1", artifact.token)

    def discard(self) -> None:
        if not self._published and not self._discarded:
            self._discard(self._effect)
            self._discarded = True

    def rollback(self) -> None:
        if self._discarded:
            return
        if self._published:
            self._rollback(self._effect, self._values)
        else:
            self._discard(self._effect)
        self._published = False
        self._discarded = True

class _PreparedLiveVisualization(PreparedPublication):
    """Compensatable intent whose irreversible frame is submitted only from ``finalize``."""

    def __init__(
        self,
        effect: AcceptedSideEffect,
        frame: _DetachedObserverFrame | None,
        submit: Any,
        journal: Any = None,
        journal_record: Any = None,
        *,
        size: int = 1,
    ) -> None:
        if isinstance(size, bool) or type(size) is not int or size < 1:
            raise TypeError("live-visualization intent size must be an integer >= 1")
        self._effect = effect
        self._frame = frame
        self._submit = submit
        self._journal = journal
        self._journal_record = journal_record
        self._size = size
        self._published = False
        self._discarded = False
        self._finalized = False

    def _discard_prepared_journal(self) -> None:
        if self._journal is None or self._journal_record is None:
            return
        if getattr(self._journal_record, "state", None) == "prepared":
            self._journal.discard_prepared(self._journal_record)

    @property
    def effect_identity(self) -> Identity:
        return self._effect.identity

    @property
    def payload_identity(self) -> Identity:
        return self._effect.payload.identity

    def publish(self) -> PublicationReceipt:
        if self._discarded:
            raise RuntimeError("discarded live-visualization intent cannot be published")
        self._published = True
        artifact = make_identity("live-visualization-intent", {
            "effect": self.effect_identity.token,
            "payload": self.payload_identity.token,
        })
        rank_artifacts = ()
        if self._effect.target.parallel_mode is ParallelMode.PER_RANK:
            rank_artifacts = tuple(
                (rank, make_identity("live-visualization-rank-intent", {
                    "intent": artifact.token,
                    "rank": rank,
                    "size": self._size,
                }).token)
                for rank in range(self._size)
            )
        return PublicationReceipt(
            self.effect_identity,
            self.payload_identity,
            "pops.live-visualization-intent.v1",
            artifact.token,
            parallel_mode=self._effect.target.parallel_mode,
            rank_artifacts=rank_artifacts,
        )

    def discard(self) -> None:
        if not self._published and not self._discarded:
            self._discard_prepared_journal()
            self._frame = None
            self._discarded = True

    def rollback(self) -> None:
        if self._finalized:
            raise RuntimeError("a submitted live frame is post-commit and cannot be rolled back")
        self._discard_prepared_journal()
        self._frame = None
        self._published = False
        self._discarded = True

    def finalize(self) -> None:
        if self._finalized:
            return None
        if not self._published or self._discarded:
            raise RuntimeError("only a published live-visualization intent can be finalized")
        # Set this boundary before dispatch so an operational finalizer retry can never duplicate
        # an irreversible packet. Journal commit/enqueue consensus is owned by the callback because
        # every MPI rank must arm its worker job together.
        preexisting_committed = (
            self._journal_record is not None
            and self._journal_record.state in {"pending", "delivered"}
        )
        self._finalized = True
        frame, self._frame = self._frame, None
        self._submit(
            self._effect,
            frame,
            self._journal,
            self._journal_record,
            preexisting_committed,
        )
        return None


class _PreparedScientificOutput(PreparedPublication):
    """One atomic output publication carrying its embedded diagnostic reductions."""

    def __init__(self, output: PreparedPublication, diagnostic: _PreparedDiagnostic) -> None:
        if output.effect_identity != diagnostic.effect_identity \
                or output.payload_identity != diagnostic.payload_identity:
            raise ValueError("scientific output and diagnostics prepare different effects")
        self._output = output
        self._diagnostic = diagnostic
        self._published = self._discarded = False

    @property
    def effect_identity(self) -> Identity:
        return self._output.effect_identity

    @property
    def payload_identity(self) -> Identity:
        return self._output.payload_identity

    @property
    def recoveries(self) -> tuple[Any, ...]:
        return self._output.recoveries

    def publish(self) -> PublicationReceipt:
        if self._discarded:
            raise RuntimeError("discarded scientific output cannot be published")
        if self._published:
            raise RuntimeError("scientific output publication is not repeatable")
        output_receipt = self._output.publish()
        try:
            self._diagnostic.publish()
        except BaseException:
            self._output.rollback()
            self._diagnostic.rollback()
            self._discarded = True
            raise
        self._published = True
        return output_receipt

    def discard(self) -> None:
        if self._published:
            self.rollback()
            return
        if not self._discarded:
            try:
                self._output.discard()
            finally:
                self._diagnostic.discard()
                self._discarded = True

    def rollback(self) -> None:
        if self._discarded:
            return
        error = None
        try:
            self._diagnostic.rollback()
        except BaseException as caught:
            error = caught
        try:
            self._output.rollback()
        except BaseException as caught:
            if error is None:
                error = caught
            else:
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    add_note("scientific output rollback also failed: %s" % caught)
        self._published = False
        self._discarded = True
        if error is not None:
            raise error

    def finalize(self) -> None:
        if not self._published or self._discarded:
            raise RuntimeError("only a published scientific output can be finalized")
        error = None
        try:
            if self._output.finalize() is not None:
                raise TypeError("scientific output finalize() must return None")
        except BaseException as caught:
            error = caught
        try:
            if self._diagnostic.finalize() is not None:
                raise TypeError("diagnostic finalize() must return None")
        except BaseException as caught:
            if error is None:
                error = caught
            else:
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    add_note("diagnostic finalization also failed: %s" % caught)
        if error is not None:
            raise error
        return None


class _PreparedCheckpoint(PreparedPublication):
    def __init__(self, effect: AcceptedSideEffect, engine: Any, operation: Any,
                 target: Any) -> None:
        self._effect, self._target, self._operation = effect, Path(target), operation
        # ``snapshot`` is the same collective prepared transaction used by
        # RuntimeInstance.checkpoint(); it captures now but remains unpublished and compensatable.
        self._snapshot = operation.snapshot(engine, self._target.parent)
        operation.validate_snapshot(self._snapshot)
        self._published = self._discarded = False

    @property
    def effect_identity(self) -> Identity:
        return self._effect.identity

    @property
    def payload_identity(self) -> Identity:
        return self._effect.payload.identity

    def publish(self) -> PublicationReceipt:
        if self._discarded:
            raise RuntimeError("discarded checkpoint cannot be published")
        if not self._published:
            produced = Path(self._operation.write(self._snapshot, self._target))
            from pops.output._checkpoint_collective import canonical_checkpoint_path

            if produced != canonical_checkpoint_path(self._target):
                raise RuntimeError("checkpoint codec published a different shared target")
            self._target = produced
            self._published = True
        artifact = make_identity("restart-checkpoint-artifact", {
            "effect": self.effect_identity.token, "target": str(self._target)})
        return PublicationReceipt(
            self.effect_identity, self.payload_identity,
            "pops.restart-checkpoint.v3", artifact.token,
            self._effect.target.parallel_mode)

    def discard(self) -> None:
        if not self._published and not self._discarded:
            self._snapshot.discard()
            self._discarded = True

    def rollback(self) -> None:
        if self._discarded:
            return
        self._snapshot.rollback()
        self._published = False
        self._discarded = True


def _writer_snapshot_data(snapshot: OutputSnapshot, request: OutputRequest) -> dict[str, Any]:
    """Project the complete selected snapshot into the generated Writer POD vocabulary."""
    import numpy as np

    fields = snapshot.select(request)
    diagnostics = snapshot.select_diagnostics(request)
    geometry_keys = {
        (field.key.layout_identity.token, field.key.level) for field in fields
    }
    diagnostic_geometry_keys = {
        (diagnostic.key.layout_identity.token, diagnostic.key.level)
        for diagnostic in diagnostics
    }
    geometries = tuple(
        geometry for geometry in snapshot.geometries
        if geometry.key in geometry_keys
        or geometry.key in diagnostic_geometry_keys
    )
    if not geometries:
        raise ValueError("native Writer snapshot has no geometry for its exact selection")
    geometry_rows = []
    for geometry in geometries:
        dimension = len(geometry.cell_shape)
        patch_identity = make_identity("writer-geometry-domain", {
            "layout": geometry.layout_identity.token,
            "level": geometry.level,
            "boxes": [list(box) for box in geometry.boxes],
        }).token
        geometry_rows.append({
            "layout_identity": geometry.layout_identity.token,
            "layout_kind": geometry.layout_kind,
            "level": geometry.level,
            "dimension": dimension,
            "patch_identity": patch_identity,
            "origin": geometry.origin,
            "spacing": geometry.spacing,
            "cell_shape": geometry.cell_shape,
            "boxes": [
                {"lower": tuple(box[:dimension]), "upper": tuple(box[dimension:])}
                for box in geometry.boxes
            ],
            # LevelGeometry owns exact, immutable C-contiguous ABI buffers.  Keep the borrowed
            # arrays intact: the generated native marshaller validates dtype/shape again.
            "valid_cells": geometry.valid_cells,
            "coverage": geometry.coverage,
            "cell_volumes": geometry.cell_volumes,
        })
    field_rows = []
    for field in fields:
        # Serial Writer v1 receives every piece.  FieldPayload has already authenticated bounds,
        # dtype and non-overlap; the C++ Writer ABI additionally proves exact geometry coverage.
        # Densifying here only to repeat that proof was an O(N) allocation on every publication.
        pieces = []
        for piece in field.pieces:
            values = np.asarray(piece.values)
            if values.dtype != np.dtype(np.float64):
                raise TypeError("native Writer ABI v1 accepts only exact float64 field pieces")
            pieces.append({
                "lower": piece.lower,
                "upper": piece.upper,
                "patch_identity": make_identity("writer-field-piece", {
                    "field": field.key.identity.token,
                    "lower": list(piece.lower), "upper": list(piece.upper),
                }).token,
                "values": np.ascontiguousarray(values),
            })
        field_rows.append({
            "field_identity": field.key.identity.token,
            "reference_id": field.key.reference.qualified_id,
            "component_manifest_identity": field.key.component_manifest_identity.token,
            "layout_identity": field.key.layout_identity.token,
            "level": field.key.level,
            "state_id": field.key.state_id,
            "centering": field.centering,
            "units": field.units,
            "component_names": field.component_names,
            "dimension": len(field.global_shape),
            "global_shape": field.global_shape,
            "pieces": pieces,
        })
    diagnostic_rows = [{
        "diagnostic_identity": value.key.identity.token,
        "reference_id": value.key.reference.qualified_id,
        "component_manifest_identity": value.key.component_manifest_identity.token,
        "layout_identity": value.key.layout_identity.token,
        "level": value.key.level,
        "state_id": value.key.state_id,
        "reduction": value.key.reduction,
        "value": value.value,
        "units": value.units,
        "terms_json": json.dumps(
            {name: item.hex() for name, item in value.terms.items()},
            sort_keys=True, separators=(",", ":")),
    } for value in diagnostics]
    return {
        "geometries": geometry_rows,
        "fields": field_rows,
        "diagnostics": diagnostic_rows,
        "metadata_json": json.dumps(
            dict(snapshot.metadata), sort_keys=True, separators=(",", ":")),
        "selection_identity": request.publication_identity.token,
    }


class _PreparedExternalWriter(PreparedPublication):
    """A verified native Writer temporary owned by one consumer transaction."""

    def __init__(self, effect: AcceptedSideEffect, preparation: OutputPreparation,
                 installed: Any, execution_context: Any) -> None:
        from pops.output.provider import consumer_format_data

        if preparation.request.consumer_id != effect.consumer_id:
            raise ValueError("native Writer request identity differs from its accepted effect")
        target_format = effect.target.output_format
        if not isinstance(target_format, Mapping):
            raise TypeError("accepted native Writer target must carry a format mapping")
        if consumer_format_data(
                preparation.format, where="resolved native Writer format") != dict(target_format):
            raise ValueError("resolved native Writer format differs from its accepted target")
        mode = effect.target.parallel_mode
        if preparation.request.parallel_mode is not mode \
                or mode not in (ParallelMode.SERIAL, ParallelMode.ROOT):
            raise ValueError(
                "native Writer ABI v1 requires one SERIAL or rank-zero ROOT complete snapshot"
            )
        if mode is ParallelMode.SERIAL and preparation.communicator is not None:
            raise ValueError("SERIAL native Writer preparation cannot carry a communicator")
        if mode is ParallelMode.ROOT and (
                preparation.communicator is None or preparation.request.rank != 0):
            raise ValueError("ROOT native Writer preparation may execute only on rank zero")
        self._effect = effect
        self._parallel_mode = mode
        self._installed = installed
        self._target = Path(preparation.target)
        self._wire = _writer_snapshot_data(preparation.snapshot, preparation.request)
        self._execution = component_execution_data(execution_context)
        self._snapshot_identity = make_identity(
            "native-writer-snapshot", preparation.snapshot.to_data(preparation.request)).token
        clock = preparation.snapshot.clock
        if clock.stage != "accepted":
            raise ValueError("native Writer publishes only an accepted snapshot stage")
        interface = installed.interface.to_data()
        self._interface_uri = interface["uri"]
        self._interface_version = interface["version"]
        self._staging = _StagingAuthority.created(
            self._target, suffix=".writer-stage")
        self._temporary = self._staging.path
        self._component_published = self._temporary.with_suffix(
            self._temporary.suffix + ".component-published")
        self._request_data = {
            "snapshot": self._wire,
            "execution": self._execution,
            "temporary_path": str(self._temporary),
            # The native component owns only this private publication name.  The runtime alone
            # links the verified inode into the public target namespace, so a component rollback
            # can never remove a concurrent or pre-existing user artifact.
            "published_path": str(self._component_published),
            "snapshot_identity": self._snapshot_identity,
            "logical_time": {
                "clock_identity": clock.clock_id,
                "tick": clock.tick,
                "level": clock.level,
                "substep": clock.substep,
                "stage": clock.stage_index,
                "fraction_numerator": clock.fraction_numerator,
                "fraction_denominator": clock.fraction_denominator,
                "dt": float.fromhex(clock.dt_hex),
                "physical_time": float.fromhex(clock.time_hex),
            },
        }
        self._recoveries: list[Any] = []
        self._published = False
        self._discarded = False
        self._created_target = False
        self._finalized = False
        try:
            receipt = self._invoke("verify")
            self._staging.authenticate_path()
            if receipt["bytes_written"] != os.fstat(self._staging.fileno()).st_size:
                raise RuntimeError("native Writer verify receipt size differs from its temporary")
            if not receipt["content_digest"]:
                raise RuntimeError("native Writer verify returned no content digest")
            self._verified_receipt = dict(receipt)
        except BaseException as error:
            cleanup_errors = self._release("rollback", include_target=False)
            add_note = getattr(error, "add_note", None)
            if callable(add_note):
                for cleanup_error in cleanup_errors:
                    add_note("native Writer verification cleanup also failed: " + cleanup_error)
            self._discarded = True
            raise

    def __del__(self) -> None:
        try:
            self._staging.close()
        except BaseException:
            pass

    @property
    def effect_identity(self) -> Identity:
        return self._effect.identity

    @property
    def payload_identity(self) -> Identity:
        return self._effect.payload.identity

    @property
    def temporary(self) -> Path:
        return self._temporary

    @property
    def target(self) -> Path:
        return self._target

    @property
    def recoveries(self) -> tuple[Any, ...]:
        return tuple(self._recoveries)

    def cleanup_recoveries(self) -> None:
        failures = []
        remaining = []
        for recovery in self._recoveries:
            try:
                recovery.cleanup_restored()
            except BaseException as error:
                failures.append(_exception_text(error))
                remaining.append(recovery)
        self._recoveries = remaining
        if failures:
            raise RuntimeError(
                "native Writer recovery cleanup failed: " + "; ".join(failures))

    def _invoke(self, operation: str, request_data: Mapping[str, Any] | None = None) -> Any:
        return self._installed.native_handle._invoke_component_operation(
            self._interface_uri, self._interface_version, operation,
            self._request_data if request_data is None else request_data)

    @staticmethod
    def _redact_cleanup_paths(value: Any, replacements: Mapping[str, str]) -> Any:
        if isinstance(value, str):
            redacted = value
            for source, destination in replacements.items():
                redacted = redacted.replace(source, destination)
            return redacted
        if isinstance(value, Mapping):
            return {
                key: _PreparedExternalWriter._redact_cleanup_paths(item, replacements)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                _PreparedExternalWriter._redact_cleanup_paths(item, replacements)
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                _PreparedExternalWriter._redact_cleanup_paths(item, replacements)
                for item in value
            )
        return value

    def _invoke_cleanup(self, operation: str) -> Any:
        """Invoke native release with only private, disposable path tombstones."""
        directory = Path(tempfile.mkdtemp(
            prefix=".pops-writer-cleanup-", dir=self._temporary.parent))
        directory_flags = (
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            parent_fd = os.open(directory.parent, directory_flags)
        except BaseException:
            directory.rmdir()
            raise
        directory_fd: int | None = None
        directory_owner: tuple[int, int] | None = None
        primary: BaseException | None = None
        try:
            directory_fd = os.open(directory.name, directory_flags, dir_fd=parent_fd)
            descriptor = os.fstat(directory_fd)
            named = os.stat(directory.name, dir_fd=parent_fd, follow_symlinks=False)
            directory_owner = (int(descriptor.st_dev), int(descriptor.st_ino))
            if not stat.S_ISDIR(descriptor.st_mode) \
                    or stat.S_IMODE(descriptor.st_mode) & 0o077 \
                    or directory_owner != (int(named.st_dev), int(named.st_ino)):
                raise RuntimeError("native Writer cleanup tombstone directory is not private")
            tombstones = {
                str(self._temporary): str(directory / "temporary.detached"),
                str(self._component_published): str(directory / "component.detached"),
                str(self._target): str(directory / "target.detached"),
            }
            request_data = self._redact_cleanup_paths(self._request_data, tombstones)
            snapshot = dict(request_data["snapshot"])
            # Metadata is not required by a release callback.  Replacing it wholesale prevents
            # an author-supplied indirect target/root path from surviving string substitution.
            snapshot["metadata_json"] = json.dumps(
                {"cleanup": "detached"}, sort_keys=True, separators=(",", ":"))
            request_data["snapshot"] = snapshot
            request_data["temporary_path"] = tombstones[str(self._temporary)]
            request_data["published_path"] = tombstones[str(self._component_published)]
            return self._invoke(operation, request_data)
        except BaseException as error:
            primary = error
            raise
        finally:
            cleanup_error: BaseException | None = None
            if directory_fd is not None:
                try:
                    for name in os.listdir(directory_fd):
                        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                        if stat.S_ISDIR(current.st_mode):
                            raise RuntimeError(
                                "native Writer cleanup created a directory inside its tombstone")
                        os.unlink(name, dir_fd=directory_fd)
                    current = os.stat(
                        directory.name, dir_fd=parent_fd, follow_symlinks=False)
                    if directory_owner is None or (
                            int(current.st_dev), int(current.st_ino)) != directory_owner:
                        raise RuntimeError(
                            "native Writer cleanup replaced its private tombstone directory")
                except BaseException as error:
                    cleanup_error = error
                finally:
                    try:
                        os.close(directory_fd)
                    except BaseException as error:
                        cleanup_error = cleanup_error or error
            if cleanup_error is None:
                try:
                    os.rmdir(directory.name, dir_fd=parent_fd)
                except BaseException as error:
                    cleanup_error = error
            try:
                os.close(parent_fd)
            except BaseException as error:
                cleanup_error = cleanup_error or error
            if cleanup_error is not None:
                if primary is not None:
                    add_note = getattr(primary, "add_note", None)
                    if callable(add_note):
                        add_note("native Writer cleanup tombstone removal also failed: %s" % cleanup_error)
                else:
                    raise cleanup_error

    @staticmethod
    def _inode(path: Path) -> tuple[int, int] | None:
        try:
            value = path.lstat()
        except FileNotFoundError:
            return None
        return int(value.st_dev), int(value.st_ino)

    def _owns(self, path: Path) -> bool:
        return self._inode(path) == self._staging.owner

    def _detach_owned(self, path: Path, *, where: str) -> None:
        try:
            _StagedOutputFile._quarantine_owned_path(
                path,
                self._staging.owner,
                replaced_message=(
                    "native Writer %s path no longer names its runtime-owned inode: %s"
                    % (where, path)),
            )
        except _OutputRecoveryRequired as error:
            self._recoveries.append(error.recovery)
            raise

    def _detach_paths(self, *, include_target: bool) -> tuple[str, ...]:
        errors = []
        paths = [
            (self._temporary, "temporary"),
            (self._component_published, "component publication"),
        ]
        if include_target and self._created_target:
            paths.append((self._target, "public target"))
        for path, where in paths:
            try:
                self._detach_owned(path, where=where)
                if path == self._target:
                    self._created_target = False
            except BaseException as error:
                errors.append(_exception_text(error))
        return tuple(errors)

    def _release(self, operation: str, *, include_target: bool) -> tuple[str, ...]:
        errors = list(self._detach_paths(include_target=include_target))
        # A component callback receives these same private path strings.  It is safe only after
        # every path was detached; otherwise a callback could unlink a replacement restored by the
        # quarantine recovery protocol.
        if not errors:
            try:
                result = self._invoke_cleanup(operation)
                if result is not None:
                    raise TypeError("native Writer %s must return None" % operation)
            except BaseException as error:
                errors.append(_exception_text(error))
        try:
            self._staging.close()
        except BaseException as error:
            errors.append(_exception_text(error))
        return tuple(errors)

    def publish(self) -> PublicationReceipt:
        if self._discarded:
            raise RuntimeError("discarded native Writer preparation cannot be published")
        if not self._published:
            try:
                receipt = self._invoke("publish")
                if self._inode(self._temporary) is not None \
                        or not self._owns(self._component_published):
                    raise RuntimeError(
                        "native Writer publish did not move its verified inode into the private "
                        "publication path")
                if dict(receipt) != self._verified_receipt:
                    raise RuntimeError(
                        "native Writer publish receipt differs from verified preparation")
                os.link(self._component_published, self._target)
                self._created_target = True
                if not self._owns(self._target):
                    raise RuntimeError(
                        "native Writer public target does not name its verified staging inode")
                self._detach_owned(
                    self._component_published, where="component publication")
                self._published = True
            except BaseException as error:
                cleanup_errors = self._release("rollback", include_target=True)
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    for cleanup_error in cleanup_errors:
                        add_note("native Writer publication cleanup also failed: " + cleanup_error)
                self._published = False
                self._discarded = True
                raise
        artifact = make_identity("native-writer-artifact", {
            "component_artifact": self._installed.artifact_identity.token,
            "snapshot": self._snapshot_identity,
            "target": str(self._target),
            "content_digest": self._verified_receipt["content_digest"],
        })
        return PublicationReceipt(
            self.effect_identity, self.payload_identity,
            "pops.output.external-writer.v1", artifact.token,
            parallel_mode=self._parallel_mode,
        )

    def discard(self) -> None:
        if self._discarded:
            return
        if self._finalized:
            raise RuntimeError("finalized native Writer cannot be discarded")
        if self._published:
            self.rollback()
            return
        errors = self._release("discard", include_target=False)
        self._discarded = True
        if errors:
            raise RuntimeError("native Writer discard failed: " + "; ".join(errors))

    def rollback(self) -> None:
        if self._discarded:
            return
        if self._finalized:
            raise RuntimeError("finalized native Writer cannot be rolled back")
        errors = self._release("rollback", include_target=True)
        self._published = False
        self._discarded = True
        if errors:
            raise RuntimeError("native Writer rollback failed: " + "; ".join(errors))

    def finalize(self) -> None:
        if self._finalized:
            return None
        if not self._published or self._discarded:
            raise RuntimeError("only a published native Writer can be finalized")
        self._staging.close()
        self._finalized = True
        return None


class _PreparedRootExternalWriter(PreparedPublication):
    """Coordinate one rank-zero native Writer transaction over the exact world communicator."""

    def __init__(self, effect: AcceptedSideEffect, preparation: OutputPreparation,
                 installed: Any, execution_context: Any) -> None:
        if preparation.request.parallel_mode is not ParallelMode.ROOT:
            raise ValueError("ROOT native Writer coordinator requires a ROOT request")
        self._effect = effect
        self._communicator = require_world(preparation.communicator)
        self._rank = native_rank(self._communicator)
        self._size = native_size(self._communicator)
        if (self._rank, self._size) != (
                preparation.request.rank, preparation.request.size):
            raise ValueError("ROOT native Writer request differs from its communicator")
        self._local: _PreparedExternalWriter | None = None
        error = None
        if self._rank == 0:
            try:
                self._local = _PreparedExternalWriter(
                    effect, preparation, installed, execution_context)
            except BaseException as exc:
                error = "%s: %s" % (type(exc).__name__, exc)
        try:
            rows = self._allgather(error=error)
            self._raise_failures("prepare", rows)
        except BaseException as failure:
            # The object is not returned when preparation consensus fails, so no outer consumer
            # transaction can compensate it. Rank zero must release its verified temporary here.
            if self._local is not None:
                try:
                    self._local.discard()
                except BaseException as cleanup_error:
                    add_note = getattr(failure, "add_note", None)
                    if callable(add_note):
                        add_note(
                            "rank-zero native Writer preparation cleanup also failed: %s: %s"
                            % (type(cleanup_error).__name__, cleanup_error)
                        )
            raise

    @property
    def effect_identity(self) -> Identity:
        return self._effect.identity

    @property
    def payload_identity(self) -> Identity:
        return self._effect.payload.identity

    @property
    def temporary(self) -> Path | None:
        return None if self._local is None else self._local.temporary

    @property
    def target(self) -> Path | None:
        return None if self._local is None else self._local.target

    @property
    def recoveries(self) -> tuple[Any, ...]:
        return () if self._local is None else self._local.recoveries

    def _allgather(
        self, *, error: str | None, artifact_id: str | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        rows = allgather_value(self._communicator, {
            "rank": self._rank,
            "error": error,
            "artifact_id": artifact_id,
        })
        if len(rows) != self._size or any(
                not isinstance(row, Mapping)
                or set(row) != {"rank", "error", "artifact_id"}
                or row["rank"] != rank
                for rank, row in enumerate(rows)):
            raise RuntimeError("ROOT native Writer returned a malformed rank envelope")
        return rows

    @staticmethod
    def _raise_failures(operation: str, rows: tuple[Mapping[str, Any], ...]) -> None:
        failures = [
            "rank %d: %s" % (rank, row["error"])
            for rank, row in enumerate(rows) if row["error"] is not None
        ]
        if failures:
            raise RuntimeError(
                "ROOT native Writer %s failed: %s"
                % (operation, "; ".join(failures))
            )

    def publish(self) -> PublicationReceipt:
        artifact_id = error = None
        if self._rank == 0:
            if self._local is None:
                error = "RuntimeError: rank zero has no prepared native Writer"
            else:
                try:
                    receipt = self._local.publish()
                    if receipt.parallel_mode is not ParallelMode.ROOT:
                        raise ValueError("rank-zero native Writer returned a non-ROOT receipt")
                    artifact_id = receipt.artifact_id
                except BaseException as exc:
                    error = "%s: %s" % (type(exc).__name__, exc)
        rows = self._allgather(error=error, artifact_id=artifact_id)
        self._raise_failures("publish", rows)
        root_artifact = rows[0]["artifact_id"]
        if not isinstance(root_artifact, str) or not root_artifact \
                or any(row["artifact_id"] is not None for row in rows[1:]):
            raise RuntimeError(
                "ROOT native Writer did not authenticate exactly one rank-zero artifact"
            )
        return PublicationReceipt(
            self.effect_identity,
            self.payload_identity,
            "pops.output.external-writer.v1",
            root_artifact,
            parallel_mode=ParallelMode.ROOT,
        )

    def _cleanup(self, operation: str) -> None:
        error = None
        if self._rank == 0:
            if self._local is None:
                error = "RuntimeError: rank zero has no prepared native Writer"
            else:
                try:
                    result = getattr(self._local, operation)()
                    if result is not None:
                        raise TypeError(
                            "rank-zero native Writer %s must return None" % operation)
                except BaseException as exc:
                    error = "%s: %s" % (type(exc).__name__, exc)
        rows = self._allgather(error=error)
        self._raise_failures(operation, rows)

    def discard(self) -> None:
        self._cleanup("discard")
        return None

    def rollback(self) -> None:
        self._cleanup("rollback")
        return None

    def finalize(self) -> None:
        self._cleanup("finalize")
        return None


class RuntimeConsumerPublisher(ConsumerPublisher):
    """One publisher for diagnostics, exact outputs, monitors and restart checkpoints."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner
        rank, size, communicator = _execution_topology(owner)
        self._by_id = {row.qualified_id: row for row in owner._consumer_graph.nodes}
        self._pending: dict[str, tuple[DiagnosticPayload, ...]] = {}
        self._pending_baselines: dict[str, dict[str, float]] = {}
        self._diagnostics: dict[str, DiagnosticPayload] = {}
        self._baselines: dict[str, float] = {}
        self._rank, self._size, self._communicator = rank, size, communicator
        self._observer_queues: dict[tuple[str, str], PostCommitObserverQueue] = {}
        self._observer_lanes: dict[tuple[str, str], Any] = {}
        self._observer_workers: dict[str, PostCommitObserverWorker] = {}
        self._observer_journals: dict[tuple[str, str], Any] = {}
        self._observer_preflight_sessions: dict[str, Any] = {}
        self._observer_reports: dict[str, ObserverDeliveryReport] = {}
        self._observer_pending_failures: dict[tuple[str, str], list[str]] = {}
        self._observer_diagnostics: list[str] = []
        self._closed_observer_runs: set[str] = set()
        self._output = ConsumerOutputPublisher(
            self._resolve_output,
            retain_recoveries=owner._retain_output_recoveries,
        )
        self._external_writers: dict[str, Any] = {}
        logical_targets: dict[str, str] = {}
        self._validate_diagnostic_providers()
        builtin_catalyst = []
        for candidate in owner._consumer_graph.nodes:
            if candidate.kind is not ConsumerKind.MONITOR:
                continue
            observer = candidate.operation_data.get("observer", {})
            provider = observer.get("provider", {}) if isinstance(observer, Mapping) else {}
            if isinstance(provider, Mapping) \
                    and provider.get("provider_id") == "pops.output.catalyst-python.v1":
                builtin_catalyst.append(candidate.qualified_id)
        if len(builtin_catalyst) > 1:
            raise ValueError(
                "the built-in Catalyst provider permits one process-global pipeline per "
                "RuntimeInstance; combine pipelines in that script or install one multiplexing "
                "provider: %s" % ", ".join(sorted(builtin_catalyst)))
        self._builtin_catalyst_consumers = tuple(sorted(builtin_catalyst))
        self._builtin_catalyst_run_started = False
        from pops import interfaces
        for manifest in owner._consumer_graph.nodes:
            if manifest.kind is ConsumerKind.MONITOR:
                data = manifest.operation_data
                if data is None or data["parallel_mode"] != manifest.parallel_mode.value:
                    raise ValueError(
                        "Monitor operation and resolved parallel mode disagree at install")
                if manifest.parallel_mode is ParallelMode.SERIAL:
                    if (rank, size, communicator) != (0, 1, None):
                        raise ValueError(
                            "SERIAL post-commit consumers require a proved serial "
                            "ExecutionContext")
                elif manifest.parallel_mode in (
                        ParallelMode.ROOT, ParallelMode.PER_RANK, ParallelMode.COLLECTIVE):
                    if communicator is None:
                        raise ValueError(
                            "%s post-commit consumer requires a proved native MPI "
                            "ExecutionContext" % manifest.parallel_mode.name)
                else:
                    raise ValueError("post-commit consumer has an unsupported parallel mode")
                preopened: Any = None
                local_error = None
                try:
                    preflight = getattr(manifest.operation, "preflight", None)
                    if callable(preflight):
                        preflight(owner._execution_context)
                    # Worker-MPI sessions are opened only after their run-scoped duplicated lane
                    # exists.  SERIAL/ROOT dependencies can still fail early at bind/install.
                    if rank == 0 and manifest.parallel_mode in (
                            ParallelMode.SERIAL, ParallelMode.ROOT):
                        preopen = getattr(manifest.operation, "preopen_session", None)
                        if not callable(preopen):
                            raise TypeError(
                                "post-commit monitor operation has no preopen_session() route")
                        preopened = preopen(owner._execution_context)
                except BaseException as error:
                    local_error = _exception_text(error)
                if manifest.parallel_mode is not ParallelMode.SERIAL:
                    try:
                        _post_commit_root_consensus(
                            communicator,
                            rank=rank,
                            size=size,
                            error=local_error,
                            phase="provider session preflight",
                        )
                    except BaseException as error:
                        if preopened is not None:
                            try:
                                preopened.abort()
                            except BaseException as abort_error:
                                add_note = getattr(error, "add_note", None)
                                if callable(add_note):
                                    add_note(
                                        "preopened observer abort also failed: %s"
                                        % _exception_text(abort_error))
                        raise
                elif local_error is not None:
                    raise RuntimeError(
                        "post-commit provider session preflight failed: %s" % local_error)
                if preopened is not None:
                    self._observer_preflight_sessions[manifest.qualified_id] = preopened
                continue
            if manifest.kind is not ConsumerKind.SCIENTIFIC_OUTPUT:
                continue
            data = manifest.output_format_data
            mode = manifest.parallel_mode
            if mode is ParallelMode.SERIAL:
                if (rank, size, communicator) != (0, 1, None):
                    raise ValueError(
                        "SERIAL ScientificOutput requires a proved serial ExecutionContext")
            elif communicator is None:
                raise ValueError(
                    "%s ScientificOutput requires a proved native MPI ExecutionContext"
                    % mode.name)
            if data["parallel_mode"] != mode.value:
                raise ValueError(
                    "ScientificOutput format and resolved parallel mode disagree at install")
            logical_target = Path(manifest.target_uri).as_posix()
            previous = logical_targets.get(logical_target)
            if previous is not None:
                raise ValueError(
                    "two ScientificOutput consumers select the same logical target: %s and %s"
                    % (previous, manifest.qualified_id))
            logical_targets[logical_target] = manifest.qualified_id
            writer = manifest.output_format.writer()
            requirement_provider = getattr(
                writer, "installed_component_requirement", None)
            if not callable(requirement_provider):
                continue
            requirement = requirement_provider()
            required_keys = {
                "component_id", "component_manifest_identity", "native_interface",
            }
            if type(requirement) is not dict or set(requirement) != required_keys:
                raise TypeError(
                    "native scientific-output writer returned a malformed component requirement")
            expected = {key: data.get(key) for key in required_keys}
            if requirement != expected:
                raise ValueError(
                    "native scientific-output writer requirement differs from format evidence")
            component_id = requirement["component_id"]
            installed = owner._installed_components.get(component_id)
            if installed is None:
                raise ValueError(
                    "ScientificOutput names native Writer %r but that exact component is not "
                    "installed" % component_id)
            if installed.component_manifest.token != requirement["component_manifest_identity"]:
                raise ValueError(
                    "ScientificOutput native Writer manifest identity differs from installation")
            if installed.interface != interfaces.Writer \
                    or dict(requirement["native_interface"]) != interfaces.Writer.to_data():
                raise ValueError("ScientificOutput component does not implement exact Writer v1")
            if installed.native_handle is None:
                raise ValueError("ScientificOutput native Writer was installed but not loaded")
            self._external_writers[manifest.qualified_id] = installed

    @property
    def diagnostics(self) -> tuple[DiagnosticPayload, ...]:
        staged = [value for rows in self._pending.values() for value in rows]
        return tuple(sorted((*self._diagnostics.values(), *staged),
                            key=lambda value: value.key.identity.token))

    @property
    def accepted_diagnostics(self) -> tuple[DiagnosticPayload, ...]:
        """Last committed registry only; staged attempt values are deliberately excluded."""
        return tuple(sorted(
            self._diagnostics.values(), key=lambda value: value.key.identity.token))

    @property
    def post_commit_reports(self) -> tuple[ObserverDeliveryReport, ...]:
        """Terminal post-commit deliveries, including reports from a still-open run."""
        rows = dict(self._observer_reports)
        for observer_queue in self._observer_queues.values():
            for report in observer_queue.reports:
                rows[report.identity.token] = report
        return tuple(sorted(
            rows.values(),
            key=lambda value: (
                value.run_identity.token, value.consumer_id, value.sequence,
                value.identity.token,
            ),
        ))

    @property
    def post_commit_diagnostics(self) -> tuple[str, ...]:
        pending = tuple(
            message
            for key in sorted(self._observer_pending_failures)
            for message in self._observer_pending_failures[key]
        )
        return tuple(self._observer_diagnostics) + pending

    @property
    def live_visualization_reports(self) -> tuple[ObserverDeliveryReport, ...]:
        """Compatibility alias for :attr:`post_commit_reports`."""
        return self.post_commit_reports

    @property
    def live_visualization_diagnostics(self) -> tuple[str, ...]:
        """Compatibility alias for :attr:`post_commit_diagnostics`."""
        return self.post_commit_diagnostics

    @staticmethod
    def _observer_key(consumer_id: str, run_identity: Identity) -> tuple[str, str]:
        if type(run_identity) is not Identity or run_identity.domain != "run":
            raise TypeError("post-commit consumer requires an exact run Identity")
        return consumer_id, run_identity.token

    def _record_observer_failure(
        self, consumer_id: str, run_identity: Identity, error: BaseException,
    ) -> None:
        key = self._observer_key(consumer_id, run_identity)
        self._observer_pending_failures.setdefault(key, []).append(_exception_text(error))

    def _observer_journal(self, manifest: Any, run_identity: Identity) -> Any:
        """Resolve one consumer/rank journal without assuming a shared filesystem."""

        key = self._observer_key(manifest.qualified_id, run_identity)
        current = self._observer_journals.get(key)
        if current is not None:
            return current
        configured = getattr(manifest.operation, "durability", None)
        if configured is None:
            return None
        from pops.output._durable_journal import DurableJournal

        if type(configured) is not DurableJournal:
            raise TypeError("installed post-commit durability is not a DurableJournal")
        root = (
            configured.root
            / manifest.identity.hexdigest
            / ("rank-%06d" % self._rank)
        )
        current = DurableJournal(root, sync=configured.sync, recover=configured.recover)
        target = Path(manifest.target_uri)
        if self._owner._output_root is not None:
            target = Path(self._owner._output_root) / target
        current.bind_delivery_authority({
            "schema_version": 1,
            "consumer_id": manifest.qualified_id,
            "manifest_identity": manifest.identity.token,
            "target_uri": manifest.target_uri,
            "resolved_target": target.expanduser().resolve().as_posix(),
        })
        self._observer_journals[key] = current
        return current

    @staticmethod
    def _journal_event(record: Any) -> str:
        frame = getattr(record, "frame", None)
        if type(frame) is not ObserverFrame:
            raise TypeError("durable journal record contains no exact ObserverFrame")
        request = frame.request.to_data()
        request.pop("rank")
        return make_identity("durable-observer-event", {
            "run_identity": frame.snapshot.provenance.run_identity.to_data(),
            "clock": frame.snapshot.clock.to_data(),
            "request": request,
        }).token

    def _inspect_observer_journal(
        self,
        manifest: Any,
        journal: Any,
    ) -> tuple[tuple[Any, ...], tuple[tuple[str, ...], ...]]:
        """Authenticate replay order/state before any observer session is initialized."""

        worker_mpi = manifest.parallel_mode in (
            ParallelMode.PER_RANK, ParallelMode.COLLECTIVE)
        if worker_mpi:
            records: tuple[Any, ...] = ()
            local_events = []
            local_error = None
            try:
                records = journal.list_committed()
                seen = set()
                for record in records:
                    event = self._journal_event(record)
                    if event in seen:
                        raise RuntimeError(
                            "durable observer journal contains duplicate committed events")
                    seen.add(event)
                    local_events.append({"event": event, "state": record.state})
            except BaseException as error:
                local_error = _exception_text(error)
                records = ()
                local_events = []
            rows = allgather_value(self._communicator, {
                "rank": self._rank,
                "events": local_events,
                "error": local_error,
            })
            if len(rows) != self._size or any(
                    not isinstance(row, Mapping)
                    or set(row) != {"rank", "events", "error"}
                    or row["rank"] != owner
                    or not isinstance(row["events"], (tuple, list))
                    or (row["error"] is not None and not isinstance(row["error"], str))
                    for owner, row in enumerate(rows)):
                raise RuntimeError(
                    "durable MPI observer replay returned malformed rank evidence")
            failures = [
                "rank %d: %s" % (owner, row["error"])
                for owner, row in enumerate(rows) if row["error"] is not None
            ]
            if failures:
                raise RuntimeError(
                    "durable MPI observer journal inspection failed collectively: "
                    + "; ".join(failures))
            sequences = []
            rank_states = []
            for row in rows:
                sequence = []
                states = []
                for item in row["events"]:
                    if not isinstance(item, Mapping) or set(item) != {"event", "state"} \
                            or not isinstance(item["event"], str) \
                            or item["state"] not in {"pending", "delivered"} \
                            or item["event"] in sequence:
                        raise RuntimeError(
                            "durable MPI observer replay contains malformed event evidence")
                    sequence.append(item["event"])
                    states.append(item["state"])
                sequences.append(tuple(sequence))
                rank_states.append(tuple(states))
            if any(sequence != sequences[0] for sequence in sequences[1:]):
                raise RuntimeError(
                    "durable MPI observer journals disagree in temporal event order after a "
                    "crash; the handoff is not atomic with the numerical checkpoint")
            return records, tuple(rank_states)
        records = journal.list_pending()
        return records, (tuple("pending" for _record in records),)

    def _replay_observer_journal(
        self,
        manifest: Any,
        observer_queue: PostCommitObserverQueue,
        journal: Any,
        records: tuple[Any, ...],
        rank_states: tuple[tuple[str, ...], ...],
    ) -> None:
        """Replay authenticated pending handoffs, including prior run identities."""

        worker_mpi = manifest.parallel_mode in (
            ParallelMode.PER_RANK, ParallelMode.COLLECTIVE)
        if worker_mpi:
            for index, record in enumerate(records):
                if not any(states[index] == "pending" for states in rank_states):
                    continue
                submission = None
                enqueue_error = None
                try:
                    submission = observer_queue._prepare_detached(
                        _detach_owned_observer_frame(record.frame),
                        journal=journal,
                        journal_record=record,
                    )
                except BaseException as error:
                    enqueue_error = _exception_text(error)
                try:
                    _post_commit_root_consensus(
                        self._communicator,
                        rank=self._rank,
                        size=self._size,
                        error=enqueue_error,
                        phase="durable replay enqueue %d" % index,
                    )
                except BaseException as error:
                    if submission is not None:
                        submission.cancel(error)
                    raise
                if submission is not None:
                    submission.arm()
            return
        for record in records:
            observer_queue.submit(
                record.frame, journal=journal, journal_record=record)

    def _open_observer_session(
        self, manifest: Any, run_identity: Identity, lane: Any,
    ) -> Any:
        del run_identity
        session = self._observer_preflight_sessions.pop(manifest.qualified_id, None)
        if session is not None:
            return session
        runtime_open = getattr(manifest.operation, "open_runtime_session", None)
        if callable(runtime_open):
            runtime_configuration = {
                "target_uri": manifest.target_uri,
                "output_root": (
                    None if self._owner._output_root is None
                    else str(self._owner._output_root)
                ),
                "consumer_id": manifest.qualified_id,
            }
            if lane is not None:
                runtime_configuration["worker_communicator"] = lane
            return runtime_open(runtime_configuration, self._owner._execution_context)
        return manifest.operation.open_session(self._owner._execution_context)

    def _observer_queue(
        self,
        manifest: Any,
        run_identity: Identity,
        *,
        session: Any = None,
        recovery_run_identities: tuple[Identity, ...] = (),
    ) -> PostCommitObserverQueue:
        key = self._observer_key(manifest.qualified_id, run_identity)
        current = self._observer_queues.get(key)
        if current is not None:
            return current
        operation_data = manifest.operation_data
        if operation_data is None:
            raise RuntimeError("post-commit consumer manifest lost its operation authority")
        lane = self._observer_lanes.get(key)
        if session is None:
            session = self._open_observer_session(manifest, run_identity, lane)
        observer_run = ObserverRun(run_identity, {
            "consumer_id": manifest.qualified_id,
            "manifest_identity": manifest.identity.token,
            "bind_identity": self._owner.bind_identity.token,
        }, recovery_run_identities)
        current = PostCommitObserverQueue(
            session,
            observer_run,
            consumer_id=manifest.qualified_id,
            capacity=operation_data["queue_capacity"],
            max_attempts=operation_data["max_attempts"],
            thread_name="pops-live-%s" % manifest.identity.hexdigest[:12],
            worker_communicator=lane,
            shared_worker=self._observer_worker(run_identity),
        )
        self._observer_queues[key] = current
        return current

    def _observer_worker(self, run_identity: Identity) -> PostCommitObserverWorker:
        self._observer_key("worker", run_identity)
        current = self._observer_workers.get(run_identity.token)
        if current is None:
            current = PostCommitObserverWorker(
                thread_name="pops-post-commit-%s" % run_identity.hexdigest[:12])
            self._observer_workers[run_identity.token] = current
        return current

    def _drain_post_commit_before_hdf5(self) -> None:
        """Exclude process-global observer-library calls from synchronous HDF5 publication."""

        for key in sorted(self._observer_queues):
            self._observer_queues[key].flush()

    def begin_post_commit_consumers(self, run_identity: Identity) -> None:
        """Initialize every active post-commit session before the first consumer/step.

        A provider may allocate run-scoped state from ``ObserverRun``.  Deferring that work until
        the first scheduled frame would allow a dependency or initialization failure after the
        numerical clock had already advanced.  ROOT ranks always exchange one status envelope
        before any rank exposes a local failure.
        """

        self._observer_key("run-begin", run_identity)
        if self._builtin_catalyst_consumers:
            if self._builtin_catalyst_run_started:
                raise RuntimeError(
                    "the built-in Catalyst lifecycle permits one run in this OS process; launch "
                    "a new process for another Catalyst simulation run")
            self._builtin_catalyst_run_started = True
        manifests = tuple(sorted(
            (row for row in self._owner._consumer_graph.nodes
             if row.kind is ConsumerKind.MONITOR),
            key=lambda value: value.qualified_id,
        ))
        for manifest in manifests:
            local_error = None
            session = None
            journal = None
            replay_records: tuple[Any, ...] = ()
            replay_states: tuple[tuple[str, ...], ...] = ((),)
            worker_mpi = manifest.parallel_mode in (
                ParallelMode.PER_RANK, ParallelMode.COLLECTIVE)
            key = self._observer_key(manifest.qualified_id, run_identity)
            if worker_mpi:
                try:
                    lane_identity = "post-commit/%s/%s" % (
                        manifest.identity.token, run_identity.token)
                    self._observer_lanes[key] = \
                        self._communicator.duplicate_observer_lane(lane_identity)
                except BaseException as error:
                    local_error = _exception_text(error)
            active = self._rank == 0 or worker_mpi
            if active and local_error is None:
                try:
                    journal = self._observer_journal(manifest, run_identity)
                except BaseException as error:
                    local_error = _exception_text(error)
            # Journal construction and its exact target binding must agree before MPI ranks inspect
            # committed events.  Otherwise a healthy rank could enter replay allgather while a
            # failing rank has already left the phase.
            if manifest.parallel_mode is not ParallelMode.SERIAL:
                try:
                    _post_commit_root_consensus(
                        self._communicator,
                        rank=self._rank,
                        size=self._size,
                        error=local_error,
                        phase="journal/lane construction",
                    )
                except BaseException:
                    self._observer_lanes.pop(key, None)
                    raise
            elif local_error is not None:
                raise RuntimeError(
                    "post-commit journal construction failed: %s" % local_error)

            local_error = None
            if active and journal is not None:
                try:
                    replay_records, replay_states = self._inspect_observer_journal(
                        manifest, journal)
                except BaseException as error:
                    local_error = _exception_text(error)
            if manifest.parallel_mode is not ParallelMode.SERIAL:
                try:
                    _post_commit_root_consensus(
                        self._communicator,
                        rank=self._rank,
                        size=self._size,
                        error=local_error,
                        phase="durable journal inspection",
                    )
                except BaseException:
                    self._observer_lanes.pop(key, None)
                    raise
            elif local_error is not None:
                raise RuntimeError(
                    "post-commit journal inspection failed: %s" % local_error)

            local_error = None
            if active:
                try:
                    session = self._open_observer_session(
                        manifest, run_identity, self._observer_lanes.get(key))
                except BaseException as error:
                    local_error = _exception_text(error)
            # No worker is started until provider imports, pipeline authentication and replay
            # inspection have succeeded everywhere.
            if manifest.parallel_mode is not ParallelMode.SERIAL:
                try:
                    _post_commit_root_consensus(
                        self._communicator,
                        rank=self._rank,
                        size=self._size,
                        error=local_error,
                        phase="session construction",
                    )
                except BaseException:
                    if session is not None:
                        try:
                            session.abort()
                        except BaseException:
                            pass
                    # Do not attempt a collective free after a possibly asymmetric communicator
                    # construction failure. ObserverMpiLane deliberately leaks safely until MPI
                    # finalization in this exceptional path instead of risking a cleanup deadlock.
                    self._observer_lanes.pop(key, None)
                    raise
            elif local_error is not None:
                raise RuntimeError(
                    "post-commit session construction failed: %s" % local_error)

            local_error = None
            if manifest.qualified_id in self._builtin_catalyst_consumers:
                try:
                    _reserve_builtin_catalyst_process_lifecycle()
                except BaseException as error:
                    local_error = _exception_text(error)
            if manifest.parallel_mode is not ParallelMode.SERIAL:
                _post_commit_root_consensus(
                    self._communicator,
                    rank=self._rank,
                    size=self._size,
                    error=local_error,
                    phase="Catalyst process lifecycle reservation",
                )
            elif local_error is not None:
                raise RuntimeError(
                    "Catalyst process lifecycle reservation failed: %s" % local_error)

            recovery_run_identities = tuple(sorted({
                record.frame.snapshot.provenance.run_identity
                for record in replay_records
                if record.frame.snapshot.provenance.run_identity != run_identity
            }, key=lambda item: item.token))
            local_error = None
            if active:
                try:
                    observer_queue = self._observer_queue(
                        manifest,
                        run_identity,
                        session=session,
                        recovery_run_identities=recovery_run_identities,
                    )
                    if journal is not None:
                        self._replay_observer_journal(
                            manifest,
                            observer_queue,
                            journal,
                            replay_records,
                            replay_states,
                        )
                except BaseException as error:
                    local_error = _exception_text(error)
            if manifest.parallel_mode is not ParallelMode.SERIAL:
                _post_commit_root_consensus(
                    self._communicator,
                    rank=self._rank,
                    size=self._size,
                    error=local_error,
                    phase="session initialization/replay",
                )
            elif local_error is not None:
                raise RuntimeError(
                    "post-commit session initialization/replay failed: %s" % local_error)

    def _submit_live_visualization(
        self,
        effect: AcceptedSideEffect,
        frame: _DetachedObserverFrame | None,
        journal: Any = None,
        journal_record: Any = None,
        preexisting_committed: bool = False,
    ) -> None:
        """Commit and arm one post-commit job only after rank-identical main-thread consensus."""
        manifest = self._manifest(effect)
        raw_frame = None
        if frame is not None:
            try:
                raw_frame = _authenticated_detached_frame(frame)
            except BaseException:
                raw_frame = None
        run_identity = self._owner.last_run_identity
        if type(run_identity) is not Identity or run_identity.domain != "run":
            # Snapshot provenance is the stronger frame-local authority when available.
            run_identity = (
                None if raw_frame is None else raw_frame.snapshot.provenance.run_identity)
        active = self._rank == 0 or manifest.parallel_mode in (
            ParallelMode.PER_RANK, ParallelMode.COLLECTIVE)
        submission = None
        local_error = None
        try:
            if type(preexisting_committed) is not bool:
                raise TypeError("post-commit preexisting flag must be an exact bool")
            if type(run_identity) is not Identity or run_identity.domain != "run":
                raise RuntimeError("post-commit dispatch lost its exact run identity")
            if active:
                if frame is None or raw_frame is None:
                    raise RuntimeError("active post-commit rank has no detached observer frame")
                if raw_frame.snapshot.provenance.run_identity != run_identity:
                    raise ValueError("post-commit frame belongs to a different run")
                committed_record = journal_record
                if journal is not None:
                    if journal_record is None:
                        raise RuntimeError("durable live frame lost its journal record")
                    if not preexisting_committed:
                        committed_record = journal.commit(journal_record)
                elif journal_record is not None:
                    raise TypeError("post-commit journal record has no DurableJournal")
                already_delivered = (
                    committed_record is not None
                    and committed_record.state == "delivered"
                )
                if not preexisting_committed and not already_delivered:
                    submission = self._observer_queue(
                        manifest, run_identity)._prepare_detached(
                            frame,
                            journal=journal,
                            journal_record=committed_record,
                        )
        except BaseException as error:
            local_error = _exception_text(error)
        consensus_error = None
        if manifest.parallel_mode is not ParallelMode.SERIAL:
            try:
                _post_commit_root_consensus(
                    self._communicator,
                    rank=self._rank,
                    size=self._size,
                    error=local_error,
                    phase="post-commit journal/enqueue",
                )
            except BaseException as error:
                consensus_error = error
        elif local_error is not None:
            consensus_error = RuntimeError(local_error)
        if consensus_error is not None:
            if submission is not None:
                submission.cancel(consensus_error)
            if active and type(run_identity) is Identity and run_identity.domain == "run":
                self._record_observer_failure(
                    manifest.qualified_id, run_identity, consensus_error)
            return None
        if type(run_identity) is not Identity or run_identity.domain != "run":
            raise RuntimeError("post-commit consensus accepted no exact run identity")
        if submission is not None:
            submission.arm()
        if manifest.parallel_mode is not ParallelMode.SERIAL:
            # A Catalyst implementation may enter MPI from its worker thread even when PoPS gives
            # it a duplicated communicator.  Do not let the next AMR/native step concurrently
            # enter solver collectives on the main thread: MPICH and third-party VTK internals do
            # not guarantee progress for that cross-library ordering.  Drain the accepted live
            # frame locally, then prove every rank has left the worker lane before any rank returns
            # to the solver.  Serial observers and asynchronous scientific writers remain async.
            delivery_error = None
            try:
                self._observer_queue(manifest, run_identity).flush()
            except BaseException as error:
                delivery_error = _exception_text(error)
            try:
                _post_commit_root_consensus(
                    self._communicator,
                    rank=self._rank,
                    size=self._size,
                    error=delivery_error,
                    phase="collective live delivery",
                )
            except BaseException as error:
                self._record_observer_failure(
                    manifest.qualified_id, run_identity, error)
        return None

    def _drain_observer_manifest(
        self,
        manifest: Any,
        run_identity: Identity,
        *,
        close: bool,
    ) -> tuple[str, ...]:
        key = self._observer_key(manifest.qualified_id, run_identity)
        local_reports: tuple[ObserverDeliveryReport, ...] = ()
        local_diagnostics = list(self._observer_pending_failures.pop(key, ()))
        worker_mpi = manifest.parallel_mode in (
            ParallelMode.PER_RANK, ParallelMode.COLLECTIVE)
        active = self._rank == 0 or worker_mpi
        observer_queue = self._observer_queues.get(key) if active else None
        if observer_queue is not None:
            try:
                local_reports = observer_queue.close() if close else observer_queue.flush()
            except BaseException as error:
                local_reports = observer_queue.reports
                local_diagnostics.append(_exception_text(error))
            finally:
                if close:
                    self._observer_queues.pop(key, None)
        if close and worker_mpi:
            lane = self._observer_lanes.pop(key, None)
            if lane is None:
                local_diagnostics.append("worker MPI lane disappeared before collective close")
            else:
                try:
                    lane.close_collectively()
                except BaseException as error:
                    local_diagnostics.append(
                        "worker MPI lane close failed: %s" % _exception_text(error))
        envelope = {
            "rank": self._rank,
            "reports": [report.to_collective_data() for report in local_reports],
            "diagnostics": local_diagnostics,
        }
        if manifest.parallel_mode is ParallelMode.ROOT:
            if self._communicator is None:
                raise RuntimeError("ROOT post-commit consumer lost its native communicator")
            rows = allgather_value(self._communicator, envelope)
            if len(rows) != self._size or any(
                    not isinstance(row, Mapping)
                    or set(row) != {"rank", "reports", "diagnostics"}
                    or row["rank"] != rank
                    or not isinstance(row["reports"], (tuple, list))
                    or not isinstance(row["diagnostics"], (tuple, list))
                    for rank, row in enumerate(rows)):
                raise RuntimeError("ROOT post-commit flush returned a malformed envelope")
            if any(row["reports"] or row["diagnostics"] for row in rows[1:]):
                raise RuntimeError(
                    "ROOT post-commit delivery occurred outside rank zero")
            authoritative = rows[0]
        elif worker_mpi:
            if self._communicator is None:
                raise RuntimeError("MPI post-commit flush lost its world communicator")
            rows = allgather_value(self._communicator, envelope)
            if len(rows) != self._size or any(
                    not isinstance(row, Mapping)
                    or set(row) != {"rank", "reports", "diagnostics"}
                    or row["rank"] != owner
                    or not isinstance(row["reports"], (tuple, list))
                    or not isinstance(row["diagnostics"], (tuple, list))
                    for owner, row in enumerate(rows)):
                raise RuntimeError("MPI post-commit flush returned a malformed envelope")
            authoritative = {
                "rank": 0,
                "reports": [report for row in rows for report in row["reports"]],
                "diagnostics": [
                    "rank %d: %s" % (owner, diagnostic)
                    for owner, row in enumerate(rows)
                    for diagnostic in row["diagnostics"]
                ],
            }
        else:
            authoritative = envelope
        reports = tuple(
            ObserverDeliveryReport.from_collective_data(dict(row))
            for row in authoritative["reports"]
        )
        for report in reports:
            if report.consumer_id != manifest.qualified_id:
                raise RuntimeError("post-commit report authenticates another session")
            self._observer_reports[report.identity.token] = report
        diagnostics = tuple(str(value) for value in authoritative["diagnostics"])
        release_diagnostics = tuple(
            "frame %s writer finalization: %s" % (
                report.frame_identity.token,
                report.receipt.detail["writer_finalize_error"],
            )
            for report in reports
            if report.receipt is not None
            and report.receipt.provider_id
            == "pops.output.async-scientific-writer.v1"
            and report.receipt.detail.get("writer_finalize_error") is not None
        )
        diagnostics += release_diagnostics
        for message in diagnostics:
            rendered = "%s [%s]: %s" % (
                manifest.qualified_id, run_identity.token, message)
            if rendered not in self._observer_diagnostics:
                self._observer_diagnostics.append(rendered)
        failures = list(diagnostics)
        failures.extend(
            "frame %s: %s" % (report.frame_identity.token, report.reason)
            for report in reports if report.status == "skipped"
        )
        if manifest.operation_data["on_failure"]["action"] == "report_only":
            return ()
        return tuple(
            "%s: %s" % (manifest.qualified_id, message) for message in failures)

    def flush_live_visualizations(
        self,
        run_identity: Identity,
        *,
        close: bool = False,
        raise_on_failure: bool = True,
    ) -> tuple[ObserverDeliveryReport, ...]:
        """Drain every live consumer for one run, with ROOT consensus on the main thread."""
        self._observer_key("run-flush", run_identity)
        if close and run_identity.token in self._closed_observer_runs:
            return tuple(
                report for report in self.post_commit_reports
                if report.run_identity == run_identity)
        failures = []
        manifests = tuple(sorted(
            (row for row in self._owner._consumer_graph.nodes
             if row.kind is ConsumerKind.MONITOR),
            key=lambda value: value.qualified_id,
        ))
        for manifest in manifests:
            try:
                failures.extend(self._drain_observer_manifest(
                    manifest, run_identity, close=close))
            except BaseException as error:
                rendered = "%s: %s" % (manifest.qualified_id, _exception_text(error))
                if rendered not in self._observer_diagnostics:
                    self._observer_diagnostics.append(rendered)
                failures.append(rendered)
        if close:
            worker = self._observer_workers.pop(run_identity.token, None)
            if worker is not None:
                try:
                    worker.close()
                except BaseException as error:
                    rendered = "post-commit worker: %s" % _exception_text(error)
                    if rendered not in self._observer_diagnostics:
                        self._observer_diagnostics.append(rendered)
                    failures.append(rendered)
            self._closed_observer_runs.add(run_identity.token)
        if failures and raise_on_failure:
            raise RuntimeError(
                "post-commit consumer delivery failed at %s: %s"
                % ("run close" if close else "flush", "; ".join(failures)))
        return tuple(
            report for report in self.post_commit_reports
            if report.run_identity == run_identity)

    def flush_post_commit_consumers(
        self,
        run_identity: Identity,
        *,
        close: bool = False,
        raise_on_failure: bool = True,
    ) -> tuple[ObserverDeliveryReport, ...]:
        return self.flush_live_visualizations(
            run_identity, close=close, raise_on_failure=raise_on_failure)

    def close_live_visualizations(
        self,
        run_identity: Identity,
        *,
        raise_on_failure: bool = True,
    ) -> tuple[ObserverDeliveryReport, ...]:
        return self.flush_live_visualizations(
            run_identity, close=True, raise_on_failure=raise_on_failure)

    def diagnostic_restart_state(self) -> dict[str, Any]:
        """Return the complete last-accepted typed diagnostic registry."""
        baselines = dict(self._baselines)
        for pending in self._pending_baselines.values():
            for key, value in pending.items():
                previous = baselines.setdefault(key, value)
                if previous != value:
                    raise RuntimeError(
                        "staged conservation diagnostics disagree on their exact baseline")
        diagnostics = dict(self._diagnostics)
        staged_diagnostics: dict[str, DiagnosticPayload] = {}
        for pending in self._pending.values():
            for payload in pending:
                token = payload.key.identity.token
                previous = staged_diagnostics.get(token)
                if previous is not None and previous.to_data() != payload.to_data():
                    raise RuntimeError(
                        "staged diagnostics disagree on the latest payload for one exact key")
                staged_diagnostics[token] = payload
                diagnostics[token] = payload
        return {
            "schema_version": 2,
            "baselines": {
                key: value.hex() for key, value in sorted(baselines.items())
            },
            "diagnostics": [
                diagnostics[token].to_data() for token in sorted(diagnostics)
            ],
        }

    @staticmethod
    def validate_diagnostic_restart_state(data: Any) -> dict[str, Any]:
        required = {"schema_version", "baselines", "diagnostics"}
        if not isinstance(data, Mapping) or set(data) != required \
                or data["schema_version"] != 2 \
                or not isinstance(data["baselines"], Mapping) \
                or not isinstance(data["diagnostics"], list):
            raise ValueError("restart diagnostic registry schema is unsupported")
        baselines = {}
        for key, value in data["baselines"].items():
            if not isinstance(key, str) or not key or not isinstance(value, str):
                raise TypeError("restart diagnostic baselines must map text identities to hex")
            scalar = float.fromhex(value)
            if not math.isfinite(scalar):
                raise ValueError("restart diagnostic baseline must be finite")
            baselines[key] = scalar
        diagnostics: dict[str, DiagnosticPayload] = {}
        from pops.model import Handle

        for row in data["diagnostics"]:
            if not isinstance(row, Mapping) \
                    or set(row) != {"key", "value", "units", "terms"} \
                    or not isinstance(row["key"], Mapping) \
                    or set(row["key"]) != {
                        "reference", "component_manifest_identity", "layout_identity",
                        "level", "state_id", "reduction",
                    } \
                    or not isinstance(row["value"], str) \
                    or not isinstance(row["terms"], Mapping):
                raise TypeError("restart diagnostic payload has an unsupported shape")
            key_data = row["key"]
            if any(not isinstance(name, str) or not name or not isinstance(value, str)
                   for name, value in row["terms"].items()):
                raise TypeError("restart diagnostic terms must map text names to hex values")
            payload = DiagnosticPayload(
                DiagnosticKey(
                    Handle.from_canonical_identity(key_data["reference"]),
                    Identity.from_token(key_data["component_manifest_identity"]),
                    Identity.from_token(key_data["layout_identity"]),
                    key_data["level"],
                    key_data["state_id"],
                    key_data["reduction"],
                ),
                float.fromhex(row["value"]),
                row["units"],
                {name: float.fromhex(value) for name, value in row["terms"].items()},
            )
            if payload.to_data() != dict(row):
                raise ValueError("restart diagnostic payload is not canonical")
            token = payload.key.identity.token
            if token in diagnostics:
                raise ValueError("restart diagnostic registry contains duplicate exact keys")
            diagnostics[token] = payload
        canonical = {
            "schema_version": 2,
            "baselines": {key: value.hex() for key, value in sorted(baselines.items())},
            "diagnostics": [diagnostics[token].to_data() for token in sorted(diagnostics)],
        }
        if canonical != dict(data):
            raise ValueError("restart diagnostic registry is not canonical")
        return canonical

    def restore_diagnostic_restart_state(self, data: Any) -> None:
        canonical = self.validate_diagnostic_restart_state(data)
        baselines = {
            key: float.fromhex(value)
            for key, value in canonical["baselines"].items()
        }
        diagnostics: dict[str, DiagnosticPayload] = {}
        from pops.model import Handle

        for row in canonical["diagnostics"]:
            key_data = row["key"]
            payload = DiagnosticPayload(
                DiagnosticKey(
                    Handle.from_canonical_identity(key_data["reference"]),
                    Identity.from_token(key_data["component_manifest_identity"]),
                    Identity.from_token(key_data["layout_identity"]),
                    key_data["level"], key_data["state_id"], key_data["reduction"],
                ),
                float.fromhex(row["value"]), row["units"],
                {name: float.fromhex(value) for name, value in row["terms"].items()},
            )
            diagnostics[payload.key.identity.token] = payload
        recorder = getattr(self._owner._executor, "record_program_diagnostic", None)
        if diagnostics and not callable(recorder):
            raise RuntimeError(
                "installed runtime cannot restore the accepted diagnostic inspection registry")
        for payload in diagnostics.values():
            cast(Any, recorder)(_diagnostic_record_name(payload), payload.value)
        self._baselines = baselines
        self._diagnostics = diagnostics
        self._pending.clear()
        self._pending_baselines.clear()

    def _manifest(self, effect: AcceptedSideEffect) -> Any:
        try:
            manifest = self._by_id[effect.consumer_id]
        except KeyError:
            raise ValueError("accepted effect names no installed ConsumerGraph node") from None
        if manifest.identity != effect.manifest_identity:
            raise ValueError("accepted effect manifest identity is stale")
        return manifest

    def _validate_diagnostic_providers(self) -> None:
        """Fail bind before execution when an exact diagnostic route is unavailable."""
        component_names = tuple(self._owner._component_manifests)
        layouts = {
            row.handle.qualified_id: row for row in self._owner._layout_plan.layouts
        }
        for manifest in self._owner._consumer_graph.nodes:
            for quantity in manifest.diagnostic_quantities:
                block = _block_name(quantity.reference, component_names)
                names, roles = _conservative_metadata(self._owner, block)
                reductions = {
                    operation["reduction"]
                    for operation in quantity.execution["operations"]
                }
                if reductions == {"step_change_l2"}:
                    if quantity.execution["role"] is not None:
                        raise ValueError("step-change norm is a whole-state diagnostic")
                    if not callable(getattr(
                            self._owner._executor_for_block(block),
                            "_step_change_l2", None)):
                        raise NotImplementedError(
                            "step-change norm requires native _step_change_l2()")
                else:
                    self._diagnostic_component(names, roles, quantity.execution["role"])
                layout = layouts.get(quantity.layout_id)
                if layout is None:
                    raise KeyError(
                        "diagnostic selected unknown layout %s" % quantity.layout_id)
                engine = self._owner._executor_for_block(block)
                if layout.adaptive:
                    if not callable(getattr(engine, "composite_reduce", None)):
                        raise NotImplementedError(
                            "adaptive diagnostic levels require native "
                            "composite_reduce(block, reduction, component, levels)"
                        )
                elif quantity.levels != (0,):
                    raise ValueError(
                        "uniform diagnostic provider accepts exactly level 0")

    def _diagnostic_metric_factor(self, quantity: Any, *, composite: bool) -> float:
        if composite:
            return 1.0
        rows = [
            row for row in self._owner._layout_plan.layouts
            if row.handle.qualified_id == quantity.layout_id
        ]
        if len(rows) != 1:
            raise KeyError("diagnostic selected unknown layout %s" % quantity.layout_id)
        geometry = rows[0].geometry
        if type(geometry) is not NormalizedGeometry:
            raise TypeError("diagnostic requires an exact normalized geometry")
        if geometry.cell_measure != CARTESIAN_CELL_AREA:
            raise NotImplementedError(
                "uniform metric-weighted diagnostics require a native provider for %s"
                % geometry.cell_measure)
        factor = 1.0
        for length, cells in zip(geometry.lengths, geometry.cells, strict=True):
            factor *= float(length) / int(cells)
        return factor

    @staticmethod
    def _diagnostic_component(
        names: tuple[str, ...], roles: tuple[str, ...], role: Any,
    ) -> tuple[int, bool]:
        if role is None:
            if len(names) != 1:
                raise ValueError(
                    "a scalar diagnostic over a multi-component state requires an explicit "
                    "typed ComponentRole selector"
                )
            return 0, False
        matches = [index for index, candidate in enumerate(roles) if candidate == role]
        if len(matches) != 1:
            raise ValueError(
                "diagnostic role %r must select exactly one conservative component; "
                "available roles are %r" % (role, roles))
        return matches[0], False

    def _native_diagnostic_reduction(
        self,
        engine: Any,
        block: str,
        reduction: str,
        component: int,
        full_state: bool,
        levels: tuple[int, ...],
    ) -> tuple[float, bool]:
        if reduction == "step_change_l2":
            if not full_state:
                raise ValueError("step-change L2 must reduce the complete conservative state")
            native = getattr(engine, "_step_change_l2", None)
            if not callable(native):
                raise RuntimeError("installed runtime has no native step-change L2 provider")
            values = native()
            if not isinstance(values, Mapping):
                raise TypeError("native step-change L2 provider returned no mapping")
            if block not in values:
                raise RuntimeError(
                    "native step-change L2 provider omitted block %r" % block)
            return float(values[block]), True
        composite = getattr(engine, "composite_reduce", None)
        if callable(composite):
            active_depth = getattr(engine, "nlev", None)
            if callable(active_depth):
                nlev = int(cast(Any, active_depth)())
                levels = tuple(level for level in levels if 0 <= int(level) < nlev)
                if not levels:
                    raise RuntimeError("adaptive diagnostic selected no active AMR level")
            kind = reduction + ("_all" if full_state else "")
            return float(cast(Any, composite)(
                block, kind, component, list(levels))), True
        if levels != (0,):
            raise ValueError("uniform diagnostic reduction accepts exactly level 0")
        native = getattr(engine, "reduce_component", None)
        if not callable(native):
            raise RuntimeError("installed runtime has no native diagnostic reduction provider")
        if full_state and reduction in {"min", "max"}:
            count = len(_conservative_names(self._owner, block))
            values = [
                float(cast(Any, native)(block, reduction, index))
                for index in range(count)
            ]
            return (min(values) if reduction == "min" else max(values)), False
        kind = reduction + ("_all" if full_state else "")
        return float(cast(Any, native)(block, kind, component)), False

    def _diagnostic_values(
        self,
        manifest: Any,
        *,
        skip_reductions: frozenset[str] = frozenset(),
    ) -> tuple[tuple[DiagnosticPayload, ...], dict[str, float]]:
        names = tuple(self._owner._component_manifests)
        values = []
        baseline_updates: dict[str, float] = {}
        for quantity in manifest.diagnostic_quantities:
            block = _block_name(quantity.reference, names)
            engine = self._owner._executor_for_block(block)
            variables, roles = _conservative_metadata(self._owner, block)
            execution = quantity.execution
            reductions = {
                operation["reduction"] for operation in execution["operations"]
            }
            if reductions == {"step_change_l2"}:
                component, full_state = 0, True
            else:
                component, full_state = self._diagnostic_component(
                    variables, roles, execution["role"])
            for operation in execution["operations"]:
                if operation["reduction"] in skip_reductions:
                    continue
                value, composite = self._native_diagnostic_reduction(
                    engine, block, operation["reduction"], component, full_state,
                    quantity.levels)
                if operation["metric_weighted"]:
                    value *= self._diagnostic_metric_factor(
                        quantity, composite=composite)
                if operation["transform"] == "sqrt":
                    if value < 0.0:
                        raise ValueError("native sum-of-squares diagnostic returned a negative value")
                    value = math.sqrt(value)
                elif operation["transform"] != "identity":
                    raise ValueError("unknown diagnostic scalar transform")
                reduction_name = operation["name"]
                terms: dict[str, float] = {}
                conservation = execution["conservation"]
                if conservation is not None:
                    baseline_key = "%s:%s" % (quantity.identity.token, reduction_name)
                    baseline = self._baselines.get(baseline_key, value)
                    drift = value - baseline
                    tolerance_token = conservation["tolerance"]
                    if not isinstance(tolerance_token, str):
                        raise TypeError(
                            "conservation diagnostic tolerance must be canonical float.hex() text")
                    try:
                        tolerance = float.fromhex(tolerance_token)
                    except (OverflowError, ValueError) as exc:
                        raise ValueError(
                            "conservation diagnostic tolerance is not valid float.hex() text"
                        ) from exc
                    if tolerance.hex() != tolerance_token \
                            or not math.isfinite(tolerance) or tolerance < 0.0:
                        raise ValueError(
                            "conservation diagnostic tolerance is not canonical finite binary64")
                    terms = {
                        "quantity": value,
                        "baseline": baseline,
                        "absolute_drift": abs(drift),
                        "tolerance": tolerance,
                    }
                    if abs(drift) > tolerance:
                        raise RuntimeError(
                            "conservation diagnostic %s drift %.17g exceeds tolerance %.17g"
                            % (quantity.handle.qualified_id, drift, tolerance))
                    baseline_updates.setdefault(baseline_key, baseline)
                    value = drift
                    reduction_name = "conservation:%s" % reduction_name
                key = DiagnosticKey(
                    quantity.handle,
                    self._owner._component_manifests[block].manifest_digest,
                    self._owner.layout_identity(quantity.layout_id),
                    min(quantity.levels) if quantity.levels else 0,
                    quantity.identity.token,
                    reduction_name,
                )
                values.append(DiagnosticPayload(key, value, "unspecified", terms))
        return tuple(values), baseline_updates

    def _publish_diagnostics(self, effect: AcceptedSideEffect,
                             values: tuple[DiagnosticPayload, ...]) -> None:
        baseline_updates = self._pending_baselines.get(effect.identity.token, {})
        for key, value in baseline_updates.items():
            self._baselines.setdefault(key, value)
        for value in values:
            self._diagnostics[value.key.identity.token] = value
            recorder = getattr(self._owner._executor, "record_program_diagnostic", None)
            if callable(recorder):
                recorder(_diagnostic_record_name(value), value.value)
        self._pending.pop(effect.identity.token, None)
        self._pending_baselines.pop(effect.identity.token, None)

    def _render_console_diagnostics(
        self,
        _effect: AcceptedSideEffect,
        manifest: Any,
        values: tuple[DiagnosticPayload, ...],
        *,
        unavailable: str | None = None,
    ) -> None:
        if self._rank != 0:
            return
        from pops.output._console_monitor import ConsoleSample

        temporal = getattr(self._owner._executor, "_temporal_restart_state", None)
        last_dt = None if temporal is None else temporal.controller_state.get(
            "last_accepted_dt")
        dt = 0.0 if last_dt is None else float.fromhex(last_dt)
        names = tuple(self._owner._component_manifests)
        sample_values: dict[str, float | None] = {}
        unavailable_values: dict[str, str] = {}
        for value in values:
            block = _block_name(value.key.reference, names)
            qualified = "%s.%s" % (block, value.key.reduction)
            sample_values[qualified] = value.value
            if len(names) == 1:
                sample_values[value.key.reduction] = value.value
                if value.key.reduction == "step_change_l2":
                    sample_values["dU_L2"] = value.value
        if unavailable is not None:
            for quantity in manifest.diagnostic_quantities:
                reductions = {
                    operation["reduction"]
                    for operation in quantity.execution["operations"]
                }
                if "step_change_l2" not in reductions:
                    continue
                block = _block_name(quantity.reference, names)
                qualified = "%s.step_change_l2" % block
                sample_values[qualified] = None
                unavailable_values[qualified] = unavailable
                if len(names) == 1:
                    sample_values["step_change_l2"] = None
                    sample_values["dU_L2"] = None
                    unavailable_values["step_change_l2"] = unavailable
                    unavailable_values["dU_L2"] = unavailable
        sample = ConsoleSample(
            time=float(self._owner._executor.time()),
            step=int(self._owner._executor.macro_step()),
            dt=dt,
            values=sample_values,
            unavailable=unavailable_values,
        )
        manifest.operation.emit(sample)

    def _discard_diagnostics(self, effect: AcceptedSideEffect) -> None:
        self._pending.pop(effect.identity.token, None)
        self._pending_baselines.pop(effect.identity.token, None)

    def _prepare_diagnostic(self, effect: AcceptedSideEffect, manifest: Any) -> Any:
        unavailable = None
        try:
            values, baseline_updates = self._diagnostic_values(manifest)
        except RuntimeError as error:
            message = str(error)
            if manifest.kind is not ConsumerKind.DIAGNOSTIC:
                raise
            if "step-change L2 unavailable after an AMR topology change" in message:
                unavailable = "AMR regrid"
            elif "step_change_l2 requires an active external step transaction" in message:
                unavailable = "initial state"
            else:
                raise
            values, baseline_updates = self._diagnostic_values(
                manifest, skip_reductions=frozenset({"step_change_l2"}))
        previous = {
            value.key.identity.token: self._diagnostics.get(value.key.identity.token)
            for value in values
        }
        existed = {
            value.key.identity.token: value.key.identity.token in self._diagnostics
            for value in values
        }
        previous_baselines = {
            key: self._baselines.get(key) for key in baseline_updates
        }
        baseline_existed = {
            key: key in self._baselines for key in baseline_updates
        }

        def rollback(_effect: AcceptedSideEffect,
                     published: tuple[DiagnosticPayload, ...]) -> None:
            for value in published:
                token = value.key.identity.token
                if existed[token]:
                    previous_value = previous[token]
                    if previous_value is None:
                        raise RuntimeError("diagnostic rollback lost its prior accepted payload")
                    self._diagnostics[token] = previous_value
                else:
                    self._diagnostics.pop(token, None)
            for key in baseline_updates:
                if baseline_existed[key]:
                    previous_value = previous_baselines[key]
                    if previous_value is None:
                        raise RuntimeError("diagnostic rollback lost its prior baseline")
                    self._baselines[key] = previous_value
                else:
                    self._baselines.pop(key, None)
            self._pending.pop(_effect.identity.token, None)
            self._pending_baselines.pop(_effect.identity.token, None)

        self._pending[effect.identity.token] = values
        self._pending_baselines[effect.identity.token] = baseline_updates
        publish_callback: Callable[
            [AcceptedSideEffect, tuple[DiagnosticPayload, ...]], None]
        if manifest.kind is ConsumerKind.DIAGNOSTIC:
            def publish_console(accepted_effect: AcceptedSideEffect,
                                accepted_values: tuple[DiagnosticPayload, ...]) -> None:
                self._publish_diagnostics(accepted_effect, accepted_values)
                self._render_console_diagnostics(
                    accepted_effect, manifest, accepted_values, unavailable=unavailable)
            publish_callback = publish_console
        else:
            publish_callback = self._publish_diagnostics
        return _PreparedDiagnostic(
            effect, values, publish_callback, self._discard_diagnostics, rollback)

    def _resolve_output(self, effect: AcceptedSideEffect) -> OutputPreparation:
        manifest = self._manifest(effect)
        if manifest.output_format_data["provider_id"] == "pops.output.hdf5.v1":
            self._drain_post_commit_before_hdf5()
        snapshot, request = self._owner._output_snapshot(
            manifest, self._pending.get(effect.identity.token, ()))
        fmt = manifest.output_format
        format_name = manifest.output_format_data["format_name"]
        target = _target(
            effect.target.uri,
            manifest.output_format_data,
            format_name,
            snapshot,
            request,
            manifest.handle.local_id, self._owner._output_root)
        _rank, _size, communicator = _execution_topology(self._owner)
        if effect.target.parallel_mode is ParallelMode.SERIAL:
            communicator = None
        return OutputPreparation(fmt, snapshot, request, target, communicator)

    def _prepare_live_visualization(
        self, effect: AcceptedSideEffect, manifest: Any,
    ) -> _PreparedLiveVisualization:
        snapshot, request = self._owner._output_snapshot(manifest)
        frame = None
        journal = None
        journal_record = None
        local_error = None
        try:
            if request.parallel_mode is not manifest.parallel_mode:
                raise RuntimeError("live-visualization snapshot parallel mode is stale")
            active = self._rank == 0 or manifest.parallel_mode in (
                ParallelMode.PER_RANK, ParallelMode.COLLECTIVE)
            if active:
                # The runtime snapshot owns its field arrays, but native geometry views are
                # borrowed.  Detach once at capture time, before the accepted native boundary can
                # regrid or advance, and carry private ownership evidence into the queue.
                frame = _detach_owned_observer_frame(ObserverFrame(snapshot, request))
                journal = self._observer_journal(
                    manifest, snapshot.provenance.run_identity)
                if journal is not None:
                    journal_record = journal.prepare(
                        _authenticated_detached_frame(frame))
        except BaseException as error:
            local_error = _exception_text(error)
        try:
            if manifest.parallel_mode is not ParallelMode.SERIAL:
                _post_commit_root_consensus(
                    self._communicator,
                    rank=self._rank,
                    size=self._size,
                    error=local_error,
                    phase="frame detachment",
                )
            elif local_error is not None:
                raise RuntimeError(
                    "post-commit frame detachment failed during durable preparation: %s"
                    % local_error)
        except BaseException:
            if journal is not None and journal_record is not None \
                    and journal_record.state == "prepared":
                journal.discard_prepared(journal_record)
            raise
        return _PreparedLiveVisualization(
            effect,
            frame,
            self._submit_live_visualization,
            journal,
            journal_record,
            size=self._size,
        )

    def prepare(self, effect: AcceptedSideEffect) -> PreparedPublication:
        if type(effect) is not AcceptedSideEffect:
            raise TypeError("RuntimeConsumerPublisher requires an exact AcceptedSideEffect")
        manifest = self._manifest(effect)
        if manifest.kind is ConsumerKind.DIAGNOSTIC:
            return self._prepare_diagnostic(effect, manifest)
        if manifest.kind is ConsumerKind.MONITOR:
            return self._prepare_live_visualization(effect, manifest)
        if manifest.kind is ConsumerKind.SCIENTIFIC_OUTPUT:
            diagnostic = self._prepare_diagnostic(effect, manifest) \
                if manifest.diagnostic_quantities else None
            try:
                installed = self._external_writers.get(manifest.qualified_id)
                if installed is not None:
                    preparation = self._resolve_output(effect)
                    if manifest.parallel_mode is ParallelMode.ROOT:
                        output = _PreparedRootExternalWriter(
                            effect, preparation, installed,
                            self._owner._execution_context,
                        )
                    else:
                        output = _PreparedExternalWriter(
                            effect, preparation, installed,
                            self._owner._execution_context)
                else:
                    output = self._output.prepare(effect)
            except BaseException:
                if diagnostic is not None:
                    diagnostic.discard()
                raise
            return output if diagnostic is None else _PreparedScientificOutput(
                output, diagnostic)
        if manifest.kind is ConsumerKind.CHECKPOINT:
            target = Path(effect.target.uri)
            if self._owner._output_root is not None:
                target = Path(self._owner._output_root) / target.name
            extension = manifest.operation_data["extension"]
            if target.suffix != extension:
                target = target.with_suffix(extension)
            return _PreparedCheckpoint(
                effect, self._owner, manifest.operation, target)
        raise TypeError("unsupported ConsumerKind %r" % manifest.kind)


class RuntimeOutputSnapshot:
    """Expose exact output values from one accepted RuntimeInstance snapshot."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner
        self._geometry_cache: dict[tuple[str, int, int], LevelGeometry] = {}

    @staticmethod
    def _native_composite_integral(
        entry: Mapping[str, Any], key: FieldKey,
    ) -> _NativeCompositeIntegral | None:
        """Invoke one preflighted native route for its exact selected level tuple."""
        reduction_method = entry["reduction_method"]
        if reduction_method is None:
            return None
        family_identity = _field_family_identity(key)
        levels = entry["reduction_levels"]
        reducer = getattr(entry["native_engine"], reduction_method)
        return _NativeCompositeIntegral(
            family_identity, levels, float(reducer(*entry["reduction_args"])))

    def _layout(self, layout_id: str) -> Any:
        rows = [row for row in self._owner._layout_plan.layouts
                if row.handle.qualified_id == layout_id]
        if len(rows) != 1:
            raise KeyError("consumer selected unknown layout %s" % layout_id)
        return rows[0]

    def _geometry(self, layout: Any, level: int) -> LevelGeometry:
        engine = self._owner._executor_for_layout(layout.handle.qualified_id)
        native_engine = getattr(engine, "_s", None)
        native_geometry = getattr(native_engine, "_output_geometry_snapshot", None)
        if not callable(native_geometry):
            raise RuntimeError(
                "scientific output requires the native output-geometry provider"
            )
        geometry = layout.geometry
        if type(geometry) is not NormalizedGeometry:
            raise TypeError("runtime output requires an exact normalized layout geometry")
        if geometry.dimension != 2:
            raise NotImplementedError(
                "the installed scientific-output provider supports rank-2 geometry; "
                "the normalized geometry has rank %d" % geometry.dimension)
        base_nx, base_ny = geometry.cells
        if int(engine.nx()) != base_nx:
            raise ValueError(
                "runtime x cell count does not match normalized layout geometry")
        if int(engine.ny()) != base_ny:
            raise ValueError(
                "runtime y cell count does not match normalized layout geometry")
        scale = layout.levels[level].refinement
        nx, ny = base_nx * scale, base_ny * scale
        if geometry.cell_measure not in _NATIVE_CELL_MEASURES:
            raise NotImplementedError(
                "scientific output does not implement normalized cell measure %s"
                % geometry.cell_measure
            )
        epoch_provider = getattr(native_engine, "checkpoint_topology_epoch", None)
        topology_epoch = int(cast(Any, epoch_provider())) \
            if layout.adaptive and callable(epoch_provider) else 0
        layout_identity = _layout_identity(layout)
        cache_key = (layout_identity.token, level, topology_epoch)
        cached = self._geometry_cache.get(cache_key)
        if cached is not None:
            return cached
        spacing = (geometry.lengths[0] / nx, geometry.lengths[1] / ny)
        next_ratio = 0
        if layout.adaptive and level + 1 < len(layout.levels):
            next_ratio = (
                layout.levels[level + 1].refinement
                // layout.levels[level].refinement
            )
        if layout.adaptive:
            native = cast(Mapping[str, Any], native_geometry(
                level, geometry.lower, spacing, (ny, nx), next_ratio,
                geometry.cell_measure))
        else:
            native = cast(Mapping[str, Any], native_geometry(
                geometry.lower, spacing, (ny, nx), geometry.cell_measure))
        if int(native["topology_epoch"]) != topology_epoch:
            raise RuntimeError("native output geometry changed during snapshot construction")
        native_boxes = tuple(
            cast(tuple[int, int, int, int], tuple(int(item) for item in box))
            for box in native["boxes"]
        )
        result = LevelGeometry(
            layout_identity, "amr" if layout.adaptive else "uniform", level,
            cast(tuple[float, float], geometry.lower), spacing, (ny, nx),
            native_boxes,
            native["coverage"], native["cell_volumes"],
            coordinate_system=geometry.coordinate_system,
            cell_measure=geometry.cell_measure,
            axis_names=cast(tuple[str, str], geometry.axis_names),
            _native_valid_cells=native["valid_cells"],
            _native_arrays=_NATIVE_GEOMETRY_ARRAYS)
        # Retain only the current topology for this qualified level.  Regridding therefore cannot
        # grow the cache indefinitely, while every quantity in one accepted epoch shares buffers.
        for stale in tuple(self._geometry_cache):
            if stale[:2] == cache_key[:2] and stale != cache_key:
                del self._geometry_cache[stale]
        self._geometry_cache[cache_key] = result
        return result

    @staticmethod
    def _local_pieces(
        native_engine: Any,
        method_name: str,
        args: tuple[Any, ...],
        *,
        mode: ParallelMode,
        rank: int,
        require_local_owner: bool = True,
    ) -> tuple[ArrayPiece, ...]:
        """Consume the exact native rank-owned output-piece ABI without reconstruction."""
        import numpy as np

        method = getattr(native_engine, method_name, None)
        if not callable(method):
            raise RuntimeError(
                "installed native provider lacks required %s() output view" % method_name)
        rows = method(*args)
        if not isinstance(rows, (tuple, list)):
            raise TypeError("%s() must return an ordered sequence of piece mappings" % method_name)
        pieces = []
        indices = set()
        required = {
            "lower", "upper", "values", "global_box_index", "owner_rank", "replicated",
        }
        for position, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != required:
                raise TypeError(
                    "%s()[%d] must contain exactly %s"
                    % (method_name, position, sorted(required)))
            box_index = row["global_box_index"]
            owner_rank = row["owner_rank"]
            replicated = row["replicated"]
            if isinstance(box_index, bool) or type(box_index) is not int or box_index < 0:
                raise TypeError("native output global_box_index must be an integer >= 0")
            if box_index in indices:
                raise ValueError("native output view contains a duplicate global_box_index")
            indices.add(box_index)
            if isinstance(owner_rank, bool) or type(owner_rank) is not int or owner_rank < 0:
                raise TypeError("native output owner_rank must be an integer >= 0")
            if type(replicated) is not bool:
                raise TypeError("native output replicated must be an exact bool")
            if require_local_owner and not replicated and owner_rank != rank:
                raise ValueError("native local output piece is owned by another rank")
            if mode in (ParallelMode.ROOT, ParallelMode.COLLECTIVE) \
                    and replicated and rank != 0:
                continue
            values = np.asarray(row["values"])
            if values.dtype != np.dtype(np.float64) or not values.flags.c_contiguous:
                raise TypeError(
                    "native output pieces must expose exact C-contiguous float64 values")
            native_bounds = []
            for name in ("lower", "upper"):
                bound = row[name]
                if not isinstance(bound, (tuple, list)) or len(bound) != 2 or any(
                        type(value) is not int for value in bound):
                    raise TypeError(
                        "native output %s must be an exact integer (j, i) pair" % name)
                native_bounds.append((bound[0], bound[1]))
            lower: tuple[int, int] = native_bounds[0]
            upper: tuple[int, int] = native_bounds[1]
            pieces.append((
                box_index,
                ArrayPiece(
                    lower,
                    upper,
                    values,
                    box_index,
                    owner_rank,
                    replicated,
                ),
            ))
        pieces.sort(key=lambda item: item[0])
        return tuple(piece for _, piece in pieces)

    @staticmethod
    def _validate_piece_bounds(
        pieces: tuple[ArrayPiece, ...], boxes: tuple[tuple[int, int, int, int], ...],
        *, complete: bool, rank: int | None = None,
    ) -> None:
        active: list[ArrayPiece] = []
        covered = 0
        for piece in sorted(pieces, key=lambda value: (value.lower, value.upper)):
            jlo, ilo = piece.lower
            jhi, ihi = piece.upper
            if piece.global_box_index >= len(boxes):
                raise ValueError("native output global_box_index lies outside geometry boxes")
            if (jlo, ilo, jhi, ihi) != boxes[piece.global_box_index]:
                raise ValueError(
                    "native output piece bounds differ from its indexed geometry box")
            if rank is not None and piece.owner_rank != rank:
                raise ValueError("rank-local native output piece has a different owner_rank")
            active = [other for other in active if other.upper[0] > jlo]
            if any(not (ihi <= other.lower[1] or other.upper[1] <= ilo) for other in active):
                raise ValueError("native output pieces overlap")
            active.append(piece)
            covered += (jhi - jlo) * (ihi - ilo)
        if complete:
            expected = sum(
                (jhi - jlo) * (ihi - ilo)
                for jlo, ilo, jhi, ihi in boxes
            )
            if covered != expected:
                raise ValueError("native output pieces do not exactly cover valid geometry boxes")
            if {piece.global_box_index for piece in pieces} != set(range(len(boxes))):
                raise ValueError(
                    "native output pieces do not authenticate every global geometry box")
            if any(piece.replicated and piece.owner_rank != 0 for piece in pieces):
                raise ValueError(
                    "complete native output uses a non-root replicated authority")

    @staticmethod
    def _piece_metadata(piece: ArrayPiece) -> dict[str, Any]:
        return {
            "lower": list(piece.lower),
            "upper": list(piece.upper),
            "global_box_index": piece.global_box_index,
            "owner_rank": piece.owner_rank,
            "replicated": piece.replicated,
        }

    @staticmethod
    def _validate_distributed_piece_metadata(
        rows: tuple[Mapping[str, Any], ...],
        *,
        mode: ParallelMode,
        boxes: tuple[tuple[int, int, int, int], ...],
    ) -> None:
        expected_keys = {"rank", "pieces", "error"}
        if any(not isinstance(row, Mapping) or set(row) != expected_keys for row in rows):
            raise TypeError("distributed output-piece envelope schema is not exact")
        if any(row["rank"] != rank for rank, row in enumerate(rows)):
            raise ValueError("distributed output-piece envelope rank order is invalid")
        by_index: dict[int, list[tuple[int, Mapping[str, Any]]]] = {}
        required_piece_keys = {
            "lower", "upper", "global_box_index", "owner_rank", "replicated",
        }
        for rank, row in enumerate(rows):
            pieces = row["pieces"]
            if not isinstance(pieces, (tuple, list)):
                raise TypeError("distributed output-piece metadata must be an ordered sequence")
            for piece in pieces:
                if not isinstance(piece, Mapping) or set(piece) != required_piece_keys:
                    raise TypeError("distributed output-piece metadata schema is not exact")
                index = piece["global_box_index"]
                if isinstance(index, bool) or type(index) is not int \
                        or index < 0 or index >= len(boxes):
                    raise ValueError("distributed output-piece global_box_index is invalid")
                owner = piece["owner_rank"]
                if isinstance(owner, bool) or type(owner) is not int \
                        or owner < 0 or owner >= len(rows):
                    raise ValueError("distributed output-piece owner_rank is invalid")
                if type(piece["replicated"]) is not bool:
                    raise TypeError("distributed output-piece replicated must be an exact bool")
                lower, upper = piece["lower"], piece["upper"]
                if (
                    not isinstance(lower, (tuple, list))
                    or not isinstance(upper, (tuple, list))
                    or len(lower) != 2
                    or len(upper) != 2
                    or any(
                        isinstance(value, bool) or type(value) is not int
                        for value in tuple(lower) + tuple(upper)
                    )
                ):
                    raise TypeError(
                        "distributed output-piece bounds must be exact integer pairs")
                if piece["owner_rank"] != rank:
                    raise ValueError(
                        "distributed output-piece owner differs from contributing rank")
                if tuple(lower) + tuple(upper) != boxes[index]:
                    raise ValueError(
                        "distributed output-piece bounds differ from indexed geometry box")
                by_index.setdefault(index, []).append((rank, piece))
        if set(by_index) != set(range(len(boxes))):
            raise ValueError("distributed output-piece union misses global geometry boxes")
        for index, contributors in by_index.items():
            replicated = {piece["replicated"] for _, piece in contributors}
            if replicated == {False}:
                if len(contributors) != 1:
                    raise ValueError(
                        "non-replicated global geometry box has multiple contributors")
                continue
            if replicated != {True}:
                raise ValueError(
                    "global geometry box mixes replicated and non-replicated ownership")
            ranks = tuple(rank for rank, _ in contributors)
            expected = (
                tuple(range(len(rows)))
                if mode is ParallelMode.PER_RANK else (0,)
            )
            if ranks != expected:
                raise ValueError(
                    "replicated global geometry box %d has an invalid contributor set" % index)

    def _distributed_pieces(
        self,
        native_engine: Any,
        method_name: str,
        args: tuple[Any, ...],
        *,
        mode: ParallelMode,
        rank: int,
        communicator: Any,
        boxes: tuple[tuple[int, int, int, int], ...],
        components: int,
    ) -> tuple[ArrayPiece, ...]:
        local: tuple[ArrayPiece, ...] = ()
        metadata: tuple[dict[str, Any], ...] = ()
        error = None
        selected_method = (
            method_name.replace("_local_pieces", "_root_pieces")
            if mode is ParallelMode.ROOT else method_name
        )
        try:
            local = self._local_pieces(
                native_engine,
                selected_method,
                (communicator, *args) if mode is ParallelMode.ROOT else args,
                mode=mode,
                rank=rank,
                require_local_owner=mode is not ParallelMode.ROOT,
            )
            if any(
                piece.values.ndim != 3 or piece.values.shape[0] != components
                for piece in local
            ):
                raise ValueError(
                    "native output piece component axis differs from the compiled state")
            self._validate_piece_bounds(
                local,
                boxes,
                complete=mode is ParallelMode.ROOT and rank == 0,
                rank=rank if mode is not ParallelMode.ROOT else None,
            )
            if mode is ParallelMode.ROOT and rank != 0 and local:
                raise RuntimeError("native ROOT output returned field data on a non-root rank")
            if mode is not ParallelMode.ROOT:
                metadata = tuple(self._piece_metadata(piece) for piece in local)
        except BaseException as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
        if mode is ParallelMode.ROOT:
            rows = allgather_value(communicator, {"rank": rank, "error": error})
            if any(
                not isinstance(row, Mapping)
                or set(row) != {"rank", "error"}
                or row["rank"] != owner
                for owner, row in enumerate(rows)
            ):
                raise RuntimeError("ROOT output-piece status schema/rank is invalid")
            failures = [
                "rank %d: %s" % (owner, row["error"])
                for owner, row in enumerate(rows) if row["error"] is not None
            ]
            if failures:
                raise RuntimeError("ROOT output-piece gather failed: " + "; ".join(failures))
            return local
        rows = allgather_value(communicator, {
            "rank": rank,
            "pieces": metadata,
            "error": error,
        })
        if any(
            not isinstance(row, Mapping)
            or set(row) != {"rank", "pieces", "error"}
            or row["rank"] != owner
            for owner, row in enumerate(rows)
        ):
            raise RuntimeError(
                "%s output-piece envelope schema/rank is invalid" % mode.name)
        failures = [
            "rank %d: %s" % (owner, row["error"])
            for owner, row in enumerate(rows) if row["error"] is not None
        ]
        if failures:
            raise RuntimeError(
                "%s output-piece preflight failed: %s"
                % (mode.name, "; ".join(failures)))
        self._validate_distributed_piece_metadata(rows, mode=mode, boxes=boxes)
        return local

    def build(self, manifest: Any, diagnostics: tuple[DiagnosticPayload, ...]) \
            -> tuple[OutputSnapshot, OutputRequest]:
        import numpy as np

        rank, size, communicator = _execution_topology(self._owner)
        mode = manifest.parallel_mode
        if mode is ParallelMode.SERIAL:
            if (rank, size) != (0, 1):
                raise ValueError("SERIAL output snapshot requires rank 0 / size 1")
        elif communicator is None:
            raise ValueError(
                "%s output snapshot requires a native MPI ExecutionContext" % mode.name)
        entries: list[dict[str, Any]] = []
        geometries: dict[tuple[str, int], LevelGeometry] = {}
        preflight_error = None
        preflight_schema: tuple[dict[str, Any], ...] = ()
        try:
            component_names = tuple(self._owner._component_manifests)
            from pops.problem.handles import FieldHandle

            for quantity in manifest.quantities:
                layout = self._layout(quantity.layout_id)
                levels = quantity.levels or tuple(row.index for row in layout.levels)
                native_cartesian_integral = (
                    layout.adaptive
                    and layout.geometry.cell_measure == CARTESIAN_CELL_AREA
                )
                block = _block_name(quantity.reference, component_names)
                component_manifest = self._owner._component_manifests[block].manifest_digest
                for level in levels:
                    geometry = self._geometry(layout, level)
                    geometries[geometry.key] = geometry
                    if isinstance(quantity.reference, FieldHandle):
                        plan = self._owner._install_plan.artifact.plan.field_plans.get(
                            quantity.reference.local_id)
                        if plan is None:
                            raise ValueError(
                                "scientific output field %r has no resolved install plan"
                                % quantity.reference.local_id)
                        engine = self._owner._executor_for_layout(
                            layout.handle.qualified_id)
                        method_name = "output_field_local_pieces"
                        args = (plan.native_options["provider_slot"], level)
                        components = (plan.operator.unknown.local_id,)
                        reduction_method = "composite_reduce_field" \
                            if native_cartesian_integral else None
                        reduction_args = (
                            plan.native_options["provider_slot"], "sum", 0, list(levels))
                    else:
                        engine = self._owner._executor_for_block(block)
                        method_name = "output_state_local_pieces"
                        args = (block, level)
                        components = _conservative_names(self._owner, block)
                        reduction_method = "composite_reduce" \
                            if native_cartesian_integral and len(components) == 1 else None
                        reduction_args = (block, "sum", 0, list(levels))
                    native_engine = engine._s
                    if not callable(getattr(native_engine, method_name, None)):
                        raise RuntimeError(
                            "installed native provider lacks required %s() output view"
                            % method_name)
                    if reduction_method is not None and not callable(
                            getattr(native_engine, reduction_method, None)):
                        raise RuntimeError(
                            "installed native provider lacks required %s() composite reduction"
                            % reduction_method)
                    entry = {
                        "quantity": quantity,
                        "geometry": geometry,
                        "component_manifest": component_manifest,
                        "native_engine": native_engine,
                        "method_name": method_name,
                        "args": args,
                        "components": components,
                        "reduction_method": reduction_method,
                        "reduction_args": reduction_args,
                        "reduction_levels": tuple(levels),
                    }
                    entries.append(entry)
            diagnostic_schema = []
            for quantity in manifest.diagnostic_quantities:
                layout = self._layout(quantity.layout_id)
                levels = quantity.levels or tuple(row.index for row in layout.levels)
                for level in levels:
                    geometry = self._geometry(layout, level)
                    geometries[geometry.key] = geometry
                diagnostic_schema.append({
                    "quantity": quantity.identity.token,
                    "handle": quantity.handle.qualified_id,
                    "layout": quantity.layout_id,
                    "levels": list(levels),
                    "execution": thaw_data(quantity.execution),
                })
            preflight_schema = tuple({
                "kind": "field",
                "quantity": entry["quantity"].identity.token,
                "geometry": entry["geometry"].to_data(),
                "component_manifest": entry["component_manifest"].token,
                "method": entry["method_name"],
                "args": list(entry["args"]),
                "components": list(entry["components"]),
                "reduction_method": entry["reduction_method"],
                "reduction_args": list(entry["reduction_args"]),
                "reduction_levels": list(entry["reduction_levels"]),
            } for entry in entries) + ({
                "kind": "diagnostics",
                "quantities": diagnostic_schema,
            },)
        except BaseException as exc:
            preflight_error = "%s: %s" % (type(exc).__name__, exc)
        if communicator is not None:
            rows = allgather_value(communicator, {
                "rank": rank,
                "schema": preflight_schema,
                "error": preflight_error,
            })
            if any(
                not isinstance(row, Mapping)
                or set(row) != {"rank", "schema", "error"}
                or row["rank"] != owner
                for owner, row in enumerate(rows)
            ):
                raise RuntimeError("output snapshot preflight envelope schema/rank is invalid")
            failures = [
                "rank %d: %s" % (owner, row["error"])
                for owner, row in enumerate(rows) if row["error"] is not None
            ]
            if failures:
                raise RuntimeError(
                    "output snapshot local preflight failed: " + "; ".join(failures))
            if any(row["schema"] != rows[0]["schema"] for row in rows[1:]):
                raise RuntimeError("output snapshot plan/geometry differs across ranks")
        elif preflight_error is not None:
            raise RuntimeError("output snapshot local preflight failed: " + preflight_error)

        extracted: list[tuple[dict[str, Any], tuple[ArrayPiece, ...]]] = []
        for entry in entries:
            geometry = entry["geometry"]
            if communicator is None:
                pieces = self._local_pieces(
                    entry["native_engine"], entry["method_name"], entry["args"],
                    mode=mode, rank=rank)
                if any(
                    piece.values.ndim != 3
                    or piece.values.shape[0] != len(entry["components"])
                    for piece in pieces
                ):
                    raise ValueError(
                        "native output piece component axis differs from compiled metadata")
                self._validate_piece_bounds(
                    pieces, geometry.boxes, complete=True, rank=rank)
            else:
                pieces = self._distributed_pieces(
                    entry["native_engine"], entry["method_name"], entry["args"],
                    mode=mode,
                    rank=rank,
                    communicator=communicator,
                    boxes=geometry.boxes,
                    components=len(entry["components"]),
                )
            extracted.append((entry, pieces))

        snapshot = request = None
        final_error = None
        canonical = None
        try:
            fields, keys = [], []
            native_integrals: dict[str, _NativeCompositeIntegral] = {}
            for entry, pieces in extracted:
                geometry = entry["geometry"]
                key = FieldKey(
                    entry["quantity"].reference,
                    entry["component_manifest"],
                    geometry.layout_identity,
                    geometry.level,
                    "accepted",
                )
                fields.append(FieldPayload(
                    key, "cell", "unspecified", entry["components"],
                    geometry.cell_shape, pieces, dtype=np.dtype(np.float64).str))
                keys.append(key)
                family_identity = _field_family_identity(key)
                levels = entry["reduction_levels"]
                authority_identity = _composite_integral_authority_identity(
                    family_identity, levels)
                if authority_identity.token not in native_integrals:
                    evidence = self._native_composite_integral(entry, key)
                    if evidence is not None:
                        native_integrals[authority_identity.token] = evidence
            selected_handles = {
                value.handle.qualified_id for value in manifest.diagnostic_quantities
            }
            selected_diagnostics = tuple(
                value for value in diagnostics
                if value.key.reference.qualified_id in selected_handles
                and value.key.layout_identity.token in {key[0] for key in geometries}
            )
            expected_diagnostic_count = sum(
                len(value.execution["operations"])
                for value in manifest.diagnostic_quantities
            )
            if len(selected_diagnostics) != expected_diagnostic_count:
                raise RuntimeError(
                    "scientific output did not stage every exact diagnostic operation")
            request = OutputRequest(
                manifest.qualified_id, tuple(keys), mode, rank, size,
                tuple(value.key for value in selected_diagnostics))
            engine = self._owner._executor
            logical_clock = manifest.schedule.domain.clock
            temporal = getattr(engine, "_temporal_restart_state", None)
            if temporal is None:
                raise RuntimeError(
                    "output snapshot requires accepted qualified temporal state")
            cursor = temporal.cursor_for_clock(logical_clock)
            last_dt_hex = temporal.controller_state.get("last_accepted_dt")
            accepted_dt = 0.0 if last_dt_hex is None else float.fromhex(last_dt_hex)
            run_identity = getattr(engine, "_last_run_identity", None)
            if type(run_identity) is not Identity:
                run_identity = make_identity("run", {
                    "runtime": self._owner._runtime_plan.identity.token,
                    "time": float(engine.time()).hex(),
                    "macro_step": int(engine.macro_step()),
                })
            snapshot = OutputSnapshot(
                OutputClock.at(
                    logical_clock.qualified_id, engine.time(), engine.macro_step(),
                    stage="accepted", tick=int(cursor["tick"]), level=0, substep=0,
                    stage_index=0, fraction=(1, 1), dt=accepted_dt),
                OutputProvenance(
                    self._owner._install_plan.artifact.plan.plan_identity,
                    self._owner._install_plan.bind_identity,
                    run_identity,
                    "runtime-instance-accepted-state",
                ),
                tuple(geometries.values()),
                tuple(fields),
                {
                    "consumer_graph": self._owner._consumer_graph.identity.token,
                    "runtime_plan": self._owner._runtime_plan.identity.token,
                },
                diagnostics=selected_diagnostics,
                _native_composite_integrals=tuple(native_integrals.values()),
            )
            canonical = snapshot.to_data(request)
            selection = request.to_data()
            selection.pop("rank")
            canonical["selection"] = selection
            canonical["fields"] = [dict(row, pieces=[]) for row in canonical["fields"]]
        except BaseException as exc:
            if communicator is None:
                raise
            final_error = "%s: %s" % (type(exc).__name__, exc)
        if communicator is not None:
            rows = allgather_value(communicator, {
                "rank": rank,
                "canonical": canonical,
                "error": final_error,
            })
            if any(
                not isinstance(row, Mapping)
                or set(row) != {"rank", "canonical", "error"}
                or row["rank"] != owner
                for owner, row in enumerate(rows)
            ):
                raise RuntimeError("output snapshot final envelope schema/rank is invalid")
            failures = [
                "rank %d: %s" % (owner, row["error"])
                for owner, row in enumerate(rows) if row["error"] is not None
            ]
            if failures:
                raise RuntimeError("output snapshot finalization failed: " + "; ".join(failures))
            if any(row["canonical"] != rows[0]["canonical"] for row in rows[1:]):
                raise RuntimeError("output snapshot canonical metadata differs across ranks")
        if snapshot is None or request is None:
            raise RuntimeError("output snapshot finalization returned no exact authority")
        return snapshot, request


__all__ = ["RuntimeConsumerPublisher", "RuntimeOutputSnapshot"]
