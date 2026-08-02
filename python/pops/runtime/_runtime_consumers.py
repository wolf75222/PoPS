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
    ObserverWorkerCollectiveLost,
    authenticate_observer_session,
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


class _ObserverCollectiveLost(RuntimeError):
    """The runtime cannot prove that every rank completed a control collective."""


class _ObserverWorkerLaneLost(RuntimeError):
    """A duplicated observer lane is sealed while MPI_COMM_WORLD remains usable."""


class _ObserverCollectiveRejected(RuntimeError):
    """Every rank returned valid evidence and at least one reported a local failure."""


def _observer_provider_id(operation_data: Any) -> str:
    """Read the authenticated provider id from either supported observer schema."""

    if not isinstance(operation_data, Mapping):
        raise TypeError("post-commit operation_data must be a mapping")
    observer = operation_data.get("observer")
    if not isinstance(observer, Mapping):
        raise TypeError("post-commit operation_data lost its observer authority")
    nested = observer.get("provider")
    provider_id = nested.get("provider_id") if isinstance(nested, Mapping) else None
    direct = observer.get("provider_id")
    if provider_id is None:
        provider_id = direct
    elif direct is not None and direct != provider_id:
        raise ValueError("post-commit observer provider authorities disagree")
    if not isinstance(provider_id, str) or not provider_id:
        raise TypeError("post-commit observer requires a non-empty provider_id")
    return provider_id


class _PendingObserverSession:
    """Run-qualified authority retained until a pre-queue session is aborted or transferred."""

    __slots__ = (
        "_abort_succeeded",
        "_authentication_error",
        "_worker_collective_lost",
        "consumer_id",
        "provider_id",
        "run_identity",
        "session",
        "worker_mpi",
    )

    def __init__(
        self,
        run_identity: Identity,
        consumer_id: str,
        provider_id: str,
        worker_mpi: bool,
        session: Any,
    ) -> None:
        if type(run_identity) is not Identity or run_identity.domain != "run":
            raise TypeError("pending observer session requires an exact run Identity")
        if not isinstance(consumer_id, str) or not consumer_id:
            raise TypeError("pending observer session requires a non-empty consumer id")
        if not isinstance(provider_id, str) or not provider_id:
            raise TypeError("pending observer session requires a non-empty provider id")
        if type(worker_mpi) is not bool:
            raise TypeError("pending observer session worker_mpi must be an exact bool")
        self.run_identity = run_identity
        self.consumer_id = consumer_id
        self.provider_id = provider_id
        self.worker_mpi = worker_mpi
        self.session = session
        self._abort_succeeded = False
        self._worker_collective_lost = False
        authentication_error = None
        try:
            authority = authenticate_observer_session(session)
            if authority["provider_id"] != provider_id:
                raise ValueError(
                    "observer session provider_id differs from its manifest: %r != %r"
                    % (authority["provider_id"], provider_id)
                )
            if authority["worker_mpi"] is not worker_mpi:
                raise ValueError(
                    "observer session worker_mpi differs from its resolved parallel mode"
                )
        except BaseException as error:
            authentication_error = _exception_text(error)
        self._authentication_error = authentication_error

    @property
    def authority(self) -> Any:
        return self.session.authority

    @property
    def abort_succeeded(self) -> bool:
        return self._abort_succeeded

    @property
    def worker_collective_lost(self) -> bool:
        return self._worker_collective_lost

    @property
    def authenticated(self) -> bool:
        return self._authentication_error is None

    @property
    def authentication_error(self) -> str | None:
        return self._authentication_error

    @property
    def close_authority(self) -> dict[str, str]:
        return {
            "run_identity": self.run_identity.token,
            "consumer_id": self.consumer_id,
            "provider_id": self.provider_id,
        }

    def abort(self) -> None:
        if self._abort_succeeded:
            return
        try:
            result = self.session.abort()
        except ObserverWorkerCollectiveLost:
            self._worker_collective_lost = True
            raise
        if result is not None:
            raise TypeError("observer abort() must return None")
        self._abort_succeeded = True

    def initialize(self, run: ObserverRun) -> Any:
        return self.session.initialize(run)

    def execute(self, frame: ObserverFrame) -> Any:
        return self.session.execute(frame)

    def finalize(self) -> Any:
        return self.session.finalize()


def _reserve_builtin_catalyst_process_lifecycle() -> None:
    """Reserve Catalyst's process-global initialize/finalize lifecycle exactly once."""

    global _BUILTIN_CATALYST_PROCESS_STARTED
    with _BUILTIN_CATALYST_PROCESS_LOCK:
        if _BUILTIN_CATALYST_PROCESS_STARTED:
            raise RuntimeError(
                "the built-in Catalyst lifecycle has already started in this OS process; "
                "launch a new process for another Catalyst simulation run"
            )
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
        row
        for row in artifact_model_metadata(owner._install_plan.artifact)
        if row.block_name == block
    ]
    if (
        len(rows) != 1
        or not rows[0].cons_names
        or len(rows[0].cons_roles) != len(rows[0].cons_names)
    ):
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
            key: _identity_payload(item, path="%s.%s" % (path, key)) for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [
            _identity_payload(item, path="%s[%d]" % (path, index))
            for index, item in enumerate(value)
        ]
    return value


def _layout_identity(layout: Any) -> Identity:
    return make_identity("layout", _identity_payload(layout.to_data()))


def _active_output_levels(
    owner: Any,
    layout: Any,
    selected: tuple[int, ...],
) -> tuple[int, ...]:
    """Intersect one authored AMR selection with the accepted live hierarchy depth."""
    configured = tuple(level.index for level in layout.levels)
    if configured != tuple(range(len(configured))):
        raise ValueError("scientific output requires contiguous configured levels starting at zero")
    if not layout.adaptive:
        if configured != (0,) or selected != (0,):
            raise ValueError("uniform scientific output accepts exactly configured level 0")
        return selected

    engine = owner._executor_for_layout(layout.handle.qualified_id)
    provider = getattr(engine, "n_levels", None)
    if not callable(provider):
        native = getattr(engine, "_s", None)
        provider = getattr(native, "n_levels", None)
    if not callable(provider):
        raise RuntimeError("adaptive scientific output requires the native active-level provider")
    active = provider()
    if isinstance(active, bool) or not isinstance(active, int):
        raise TypeError("native active AMR level count must be an exact integer")
    if active < 1 or active > len(configured):
        raise RuntimeError(
            "native active AMR depth %d lies outside the configured [1, %d] envelope"
            % (active, len(configured))
        )
    result = tuple(level for level in selected if level < active)
    if not result:
        raise RuntimeError(
            "scientific output selection %r contains no currently active AMR level" % (selected,)
        )
    return result


_NATIVE_CELL_MEASURES = frozenset(
    {
        CARTESIAN_CELL_AREA,
        POLAR_ANNULUS_CELL_AREA,
    }
)


def _target(
    uri: str,
    format_data: Mapping[str, Any],
    format_name: str,
    snapshot: OutputSnapshot,
    request: OutputRequest,
    consumer_name: str,
    output_root: Any,
) -> Path:
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

    try:
        rows = allgather_value(communicator, {"rank": rank, "error": error})
    except BaseException as collective_error:
        raise _ObserverCollectiveLost(
            "ROOT post-commit %s lost its collective proof: %s"
            % (phase, _exception_text(collective_error))
        ) from collective_error
    if len(rows) != size or any(
        not isinstance(row, Mapping)
        or set(row) != {"rank", "error"}
        or row["rank"] != owner_rank
        or (row["error"] is not None and not isinstance(row["error"], str))
        for owner_rank, row in enumerate(rows)
    ):
        raise _ObserverCollectiveLost("ROOT post-commit %s returned a malformed envelope" % phase)
    failures = [
        "rank %d: %s" % (owner_rank, row["error"])
        for owner_rank, row in enumerate(rows)
        if row["error"] is not None
    ]
    if failures:
        raise _ObserverCollectiveRejected(
            "ROOT post-commit %s failed: %s" % (phase, "; ".join(failures))
        )


class _PreparedDiagnostic(PreparedPublication):
    def __init__(
        self,
        effect: AcceptedSideEffect,
        values: tuple[DiagnosticPayload, ...],
        publish: Callable[[AcceptedSideEffect, tuple[DiagnosticPayload, ...]], None],
        discard: Callable[[AcceptedSideEffect], None],
        rollback: Callable[[AcceptedSideEffect, tuple[DiagnosticPayload, ...]], None],
    ) -> None:
        self._effect, self._values = effect, values
        self._publish, self._discard, self._rollback = publish, discard, rollback
        self._published = self._discarded = False

    @property
    def effect_identity(self) -> Identity:
        return self._effect.identity

    @property
    def payload_identity(self) -> Identity:
        return self._effect.payload.identity

    @property
    def recoveries(self) -> tuple[Any, ...]:
        return ()

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
        artifact = make_identity(
            "runtime-diagnostic-publication",
            {
                "effect": self.effect_identity.token,
                "values": [value.to_data() for value in self._values],
            },
        )
        return PublicationReceipt(
            self.effect_identity,
            self.payload_identity,
            "pops.runtime-diagnostic.v1",
            artifact.token,
        )

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
        artifact = make_identity(
            "live-visualization-intent",
            {
                "effect": self.effect_identity.token,
                "payload": self.payload_identity.token,
            },
        )
        rank_artifacts = ()
        if self._effect.target.parallel_mode is ParallelMode.PER_RANK:
            rank_artifacts = tuple(
                (
                    rank,
                    make_identity(
                        "live-visualization-rank-intent",
                        {
                            "intent": artifact.token,
                            "rank": rank,
                            "size": self._size,
                        },
                    ).token,
                )
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
        preexisting_committed = self._journal_record is not None and self._journal_record.state in {
            "pending",
            "delivered",
        }
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
        if (
            output.effect_identity != diagnostic.effect_identity
            or output.payload_identity != diagnostic.payload_identity
        ):
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
        recoveries = getattr(self._output, "recoveries", ())
        if not isinstance(recoveries, tuple):
            raise TypeError("prepared output recoveries must be a tuple")
        return recoveries

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
    def __init__(
        self, effect: AcceptedSideEffect, engine: Any, operation: Any, target: Any
    ) -> None:
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
        artifact = make_identity(
            "restart-checkpoint-artifact",
            {"effect": self.effect_identity.token, "target": str(self._target)},
        )
        return PublicationReceipt(
            self.effect_identity,
            self.payload_identity,
            "pops.restart-checkpoint.v5",
            artifact.token,
            self._effect.target.parallel_mode,
        )

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
    geometry_keys = {(field.key.layout_identity.token, field.key.level) for field in fields}
    diagnostic_geometry_keys = {
        (diagnostic.key.layout_identity.token, diagnostic.key.level) for diagnostic in diagnostics
    }
    geometries = tuple(
        geometry
        for geometry in snapshot.geometries
        if geometry.key in geometry_keys or geometry.key in diagnostic_geometry_keys
    )
    if not geometries:
        raise ValueError("native Writer snapshot has no geometry for its exact selection")
    geometry_rows = []
    for geometry in geometries:
        dimension = len(geometry.cell_shape)
        patch_identity = make_identity(
            "writer-geometry-domain",
            {
                "layout": geometry.layout_identity.token,
                "level": geometry.level,
                "boxes": [list(box) for box in geometry.boxes],
            },
        ).token
        geometry_rows.append(
            {
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
            }
        )
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
            pieces.append(
                {
                    "lower": piece.lower,
                    "upper": piece.upper,
                    "patch_identity": make_identity(
                        "writer-field-piece",
                        {
                            "field": field.key.identity.token,
                            "lower": list(piece.lower),
                            "upper": list(piece.upper),
                        },
                    ).token,
                    "values": np.ascontiguousarray(values),
                }
            )
        field_rows.append(
            {
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
            }
        )
    diagnostic_rows = [
        {
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
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        for value in diagnostics
    ]
    return {
        "geometries": geometry_rows,
        "fields": field_rows,
        "diagnostics": diagnostic_rows,
        "metadata_json": json.dumps(dict(snapshot.metadata), sort_keys=True, separators=(",", ":")),
        "selection_identity": request.publication_identity.token,
    }


class _PreparedExternalWriter(PreparedPublication):
    """A verified native Writer temporary owned by one consumer transaction."""

    def __init__(
        self,
        effect: AcceptedSideEffect,
        preparation: OutputPreparation,
        installed: Any,
        execution_context: Any,
    ) -> None:
        from pops.output.provider import consumer_format_data

        if preparation.request.consumer_id != effect.consumer_id:
            raise ValueError("native Writer request identity differs from its accepted effect")
        target_format = effect.target.output_format
        if not isinstance(target_format, Mapping):
            raise TypeError("accepted native Writer target must carry a format mapping")
        if consumer_format_data(preparation.format, where="resolved native Writer format") != dict(
            target_format
        ):
            raise ValueError("resolved native Writer format differs from its accepted target")
        mode = effect.target.parallel_mode
        if preparation.request.parallel_mode is not mode or mode not in (
            ParallelMode.SERIAL,
            ParallelMode.ROOT,
        ):
            raise ValueError(
                "native Writer ABI v1 requires one SERIAL or rank-zero ROOT complete snapshot"
            )
        if mode is ParallelMode.SERIAL and preparation.communicator is not None:
            raise ValueError("SERIAL native Writer preparation cannot carry a communicator")
        if mode is ParallelMode.ROOT and (
            preparation.communicator is None or preparation.request.rank != 0
        ):
            raise ValueError("ROOT native Writer preparation may execute only on rank zero")
        self._effect = effect
        self._parallel_mode = mode
        self._installed = installed
        self._target = Path(preparation.target)
        self._wire = _writer_snapshot_data(preparation.snapshot, preparation.request)
        self._execution = component_execution_data(execution_context)
        self._snapshot_identity = make_identity(
            "native-writer-snapshot", preparation.snapshot.to_data(preparation.request)
        ).token
        clock = preparation.snapshot.clock
        if clock.stage != "accepted":
            raise ValueError("native Writer publishes only an accepted snapshot stage")
        interface = installed.interface.to_data()
        self._interface_uri = interface["uri"]
        self._interface_version = interface["version"]
        self._staging = _StagingAuthority.created(self._target, suffix=".writer-stage")
        self._temporary = self._staging.path
        self._component_published = self._temporary.with_suffix(
            self._temporary.suffix + ".component-published"
        )
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
            raise RuntimeError("native Writer recovery cleanup failed: " + "; ".join(failures))

    def _invoke(self, operation: str, request_data: Mapping[str, Any] | None = None) -> Any:
        return self._installed.native_handle._invoke_component_operation(
            self._interface_uri,
            self._interface_version,
            operation,
            self._request_data if request_data is None else request_data,
        )

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
                _PreparedExternalWriter._redact_cleanup_paths(item, replacements) for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                _PreparedExternalWriter._redact_cleanup_paths(item, replacements) for item in value
            )
        return value

    def _invoke_cleanup(self, operation: str) -> Any:
        """Invoke native release with only private, disposable path tombstones."""
        directory = Path(
            tempfile.mkdtemp(prefix=".pops-writer-cleanup-", dir=self._temporary.parent)
        )
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
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
            if (
                not stat.S_ISDIR(descriptor.st_mode)
                or stat.S_IMODE(descriptor.st_mode) & 0o077
                or directory_owner != (int(named.st_dev), int(named.st_ino))
            ):
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
                {"cleanup": "detached"}, sort_keys=True, separators=(",", ":")
            )
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
                                "native Writer cleanup created a directory inside its tombstone"
                            )
                        os.unlink(name, dir_fd=directory_fd)
                    current = os.stat(directory.name, dir_fd=parent_fd, follow_symlinks=False)
                    if (
                        directory_owner is None
                        or (int(current.st_dev), int(current.st_ino)) != directory_owner
                    ):
                        raise RuntimeError(
                            "native Writer cleanup replaced its private tombstone directory"
                        )
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
                        add_note(
                            "native Writer cleanup tombstone removal also failed: %s"
                            % cleanup_error
                        )
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
                    % (where, path)
                ),
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
                if self._inode(self._temporary) is not None or not self._owns(
                    self._component_published
                ):
                    raise RuntimeError(
                        "native Writer publish did not move its verified inode into the private "
                        "publication path"
                    )
                if dict(receipt) != self._verified_receipt:
                    raise RuntimeError(
                        "native Writer publish receipt differs from verified preparation"
                    )
                os.link(self._component_published, self._target)
                self._created_target = True
                if not self._owns(self._target):
                    raise RuntimeError(
                        "native Writer public target does not name its verified staging inode"
                    )
                self._detach_owned(self._component_published, where="component publication")
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
        artifact = make_identity(
            "native-writer-artifact",
            {
                "component_artifact": self._installed.artifact_identity.token,
                "snapshot": self._snapshot_identity,
                "target": str(self._target),
                "content_digest": self._verified_receipt["content_digest"],
            },
        )
        return PublicationReceipt(
            self.effect_identity,
            self.payload_identity,
            "pops.output.external-writer.v1",
            artifact.token,
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

    def __init__(
        self,
        effect: AcceptedSideEffect,
        preparation: OutputPreparation,
        installed: Any,
        execution_context: Any,
    ) -> None:
        if preparation.request.parallel_mode is not ParallelMode.ROOT:
            raise ValueError("ROOT native Writer coordinator requires a ROOT request")
        self._effect = effect
        self._communicator = require_world(preparation.communicator)
        self._rank = native_rank(self._communicator)
        self._size = native_size(self._communicator)
        if (self._rank, self._size) != (preparation.request.rank, preparation.request.size):
            raise ValueError("ROOT native Writer request differs from its communicator")
        self._local: _PreparedExternalWriter | None = None
        error = None
        if self._rank == 0:
            try:
                self._local = _PreparedExternalWriter(
                    effect, preparation, installed, execution_context
                )
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
        self,
        *,
        error: str | None,
        artifact_id: str | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        rows = allgather_value(
            self._communicator,
            {
                "rank": self._rank,
                "error": error,
                "artifact_id": artifact_id,
            },
        )
        if len(rows) != self._size or any(
            not isinstance(row, Mapping)
            or set(row) != {"rank", "error", "artifact_id"}
            or row["rank"] != rank
            for rank, row in enumerate(rows)
        ):
            raise RuntimeError("ROOT native Writer returned a malformed rank envelope")
        return rows

    @staticmethod
    def _raise_failures(operation: str, rows: tuple[Mapping[str, Any], ...]) -> None:
        failures = [
            "rank %d: %s" % (rank, row["error"])
            for rank, row in enumerate(rows)
            if row["error"] is not None
        ]
        if failures:
            raise RuntimeError(
                "ROOT native Writer %s failed: %s" % (operation, "; ".join(failures))
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
        if (
            not isinstance(root_artifact, str)
            or not root_artifact
            or any(row["artifact_id"] is not None for row in rows[1:])
        ):
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
                        raise TypeError("rank-zero native Writer %s must return None" % operation)
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
        self._observer_queues: dict[tuple[str, str], PostCommitObserverQueue | None] = {}
        self._observer_lanes: dict[tuple[str, str], Any] = {}
        self._root_output_lanes: dict[str, Any] = {}
        self._observer_workers: dict[str, PostCommitObserverWorker | None] = {}
        self._observer_pending_sessions: dict[tuple[str, str], _PendingObserverSession | None] = {}
        self._observer_journals: dict[tuple[str, str], Any] = {}
        self._observer_preflight_sessions: dict[str, Any] = {}
        self._observer_reports: dict[str, ObserverDeliveryReport] = {}
        self._observer_pending_reports: dict[
            tuple[str, str], tuple[ObserverDeliveryReport, ...]
        ] = {}
        self._observer_report_run_authorities: dict[tuple[str, str], frozenset[Identity]] = {}
        self._observer_pending_failures: dict[tuple[str, str], list[str]] = {}
        self._observer_abort_retry_blocked: set[tuple[str, str]] = set()
        self._observer_finalize_retry_blocked: set[tuple[str, str]] = set()
        self._observer_world_collective_lost: str | None = None
        self._observer_diagnostics: list[str] = []
        self._closed_observer_runs: set[str] = set()
        self._observer_run_phases: dict[str, str] = {}
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
            if (
                isinstance(provider, Mapping)
                and provider.get("provider_id") == "pops.output.catalyst-python.v1"
            ):
                builtin_catalyst.append(candidate.qualified_id)
        if len(builtin_catalyst) > 1:
            raise ValueError(
                "the built-in Catalyst provider permits one process-global pipeline per "
                "RuntimeInstance; combine pipelines in that script or install one multiplexing "
                "provider: %s" % ", ".join(sorted(builtin_catalyst))
            )
        self._builtin_catalyst_consumers = tuple(sorted(builtin_catalyst))
        self._builtin_catalyst_run_started = False
        self._root_output_consumers = tuple(
            sorted(
                candidate.qualified_id
                for candidate in owner._consumer_graph.nodes
                if candidate.kind
                in {
                    ConsumerKind.SCIENTIFIC_OUTPUT,
                    ConsumerKind.MONITOR,
                }
                and candidate.parallel_mode is ParallelMode.ROOT
            )
        )
        from pops import interfaces

        for manifest in owner._consumer_graph.nodes:
            if manifest.kind is ConsumerKind.MONITOR:
                data = manifest.operation_data
                if data is None or data["parallel_mode"] != manifest.parallel_mode.value:
                    raise ValueError(
                        "Monitor operation and resolved parallel mode disagree at install"
                    )
                if manifest.parallel_mode is ParallelMode.SERIAL:
                    if (rank, size, communicator) != (0, 1, None):
                        raise ValueError(
                            "SERIAL post-commit consumers require a proved serial ExecutionContext"
                        )
                elif manifest.parallel_mode in (
                    ParallelMode.ROOT,
                    ParallelMode.PER_RANK,
                    ParallelMode.COLLECTIVE,
                ):
                    if communicator is None:
                        raise ValueError(
                            "%s post-commit consumer requires a proved native MPI "
                            "ExecutionContext" % manifest.parallel_mode.name
                        )
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
                        ParallelMode.SERIAL,
                        ParallelMode.ROOT,
                    ):
                        preopen = getattr(manifest.operation, "preopen_session", None)
                        if not callable(preopen):
                            raise TypeError(
                                "post-commit monitor operation has no preopen_session() route"
                            )
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
                                        % _exception_text(abort_error)
                                    )
                        raise
                elif local_error is not None:
                    raise RuntimeError(
                        "post-commit provider session preflight failed: %s" % local_error
                    )
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
                        "SERIAL ScientificOutput requires a proved serial ExecutionContext"
                    )
            elif communicator is None:
                raise ValueError(
                    "%s ScientificOutput requires a proved native MPI ExecutionContext" % mode.name
                )
            if data["parallel_mode"] != mode.value:
                raise ValueError(
                    "ScientificOutput format and resolved parallel mode disagree at install"
                )
            logical_target = Path(manifest.target_uri).as_posix()
            previous = logical_targets.get(logical_target)
            if previous is not None:
                raise ValueError(
                    "two ScientificOutput consumers select the same logical target: %s and %s"
                    % (previous, manifest.qualified_id)
                )
            logical_targets[logical_target] = manifest.qualified_id
            writer = manifest.output_format.writer()
            requirement_provider = getattr(writer, "installed_component_requirement", None)
            if not callable(requirement_provider):
                continue
            requirement = requirement_provider()
            required_keys = {
                "component_id",
                "component_manifest_identity",
                "native_interface",
            }
            if type(requirement) is not dict or set(requirement) != required_keys:
                raise TypeError(
                    "native scientific-output writer returned a malformed component requirement"
                )
            expected = {key: data.get(key) for key in required_keys}
            if requirement != expected:
                raise ValueError(
                    "native scientific-output writer requirement differs from format evidence"
                )
            component_id = requirement["component_id"]
            installed = owner._installed_components.get(component_id)
            if installed is None:
                raise ValueError(
                    "ScientificOutput names native Writer %r but that exact component is not "
                    "installed" % component_id
                )
            if installed.component_manifest.token != requirement["component_manifest_identity"]:
                raise ValueError(
                    "ScientificOutput native Writer manifest identity differs from installation"
                )
            if (
                installed.interface != interfaces.Writer
                or dict(requirement["native_interface"]) != interfaces.Writer.to_data()
            ):
                raise ValueError("ScientificOutput component does not implement exact Writer v1")
            if installed.native_handle is None:
                raise ValueError("ScientificOutput native Writer was installed but not loaded")
            self._external_writers[manifest.qualified_id] = installed

    @property
    def diagnostics(self) -> tuple[DiagnosticPayload, ...]:
        staged = [value for rows in self._pending.values() for value in rows]
        return tuple(
            sorted(
                (*self._diagnostics.values(), *staged), key=lambda value: value.key.identity.token
            )
        )

    @property
    def accepted_diagnostics(self) -> tuple[DiagnosticPayload, ...]:
        """Last committed registry only; staged attempt values are deliberately excluded."""
        return tuple(sorted(self._diagnostics.values(), key=lambda value: value.key.identity.token))

    @property
    def post_commit_reports(self) -> tuple[ObserverDeliveryReport, ...]:
        """Deliveries authenticated by the run's required main-thread consensus."""
        rows = dict(self._observer_reports)
        return tuple(
            sorted(
                rows.values(),
                key=lambda value: (
                    value.run_identity.token,
                    value.consumer_id,
                    value.sequence,
                    value.identity.token,
                ),
            )
        )

    @property
    def post_commit_diagnostics(self) -> tuple[str, ...]:
        pending = tuple(
            message
            for key in sorted(self._observer_pending_failures)
            for message in self._observer_pending_failures[key]
        )
        return tuple(self._observer_diagnostics) + pending

    def seal_observer_collective_loss(self, error: BaseException) -> bool:
        """Seal WORLD-backed observer operations when an exception chain lost their proof."""

        if getattr(self, "_observer_world_collective_lost", None) is not None:
            return True
        pending: list[BaseException] = [error]
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            if isinstance(current, _ObserverCollectiveLost):
                if getattr(self, "_observer_world_collective_lost", None) is None:
                    self._observer_world_collective_lost = _exception_text(current)
                return True
            if current.__cause__ is not None:
                pending.append(current.__cause__)
            if current.__context__ is not None:
                pending.append(current.__context__)
        return False

    def seal_observer_workers_after_world_loss(self, error: BaseException) -> tuple[str, ...]:
        """Stop local non-daemon workers without MPI or provider lifecycle re-entry."""

        if not isinstance(error, BaseException):
            raise TypeError("observer WORLD-loss sealing requires an exception")
        if getattr(self, "_observer_world_collective_lost", None) is None:
            raise RuntimeError("observer workers may be sealed only after WORLD proof loss")
        local_error = RuntimeError(
            "post-commit worker sealed locally after MPI_COMM_WORLD collective proof loss"
        )
        failures: list[str] = []
        for key in sorted(getattr(self, "_observer_queues", {})):
            observer_queue = self._observer_queues[key]
            if observer_queue is None:
                continue
            seal_local = getattr(observer_queue, "seal_local", None)
            if not callable(seal_local):
                failures.append("observer queue %r has no local seal route" % (key,))
                continue
            try:
                seal_local(local_error)
            except BaseException as caught:
                failures.append(
                    "observer queue %r local seal failed: %s" % (key, _exception_text(caught))
                )
        for run_key in sorted(getattr(self, "_observer_workers", {})):
            worker = self._observer_workers[run_key]
            if worker is None:
                continue
            seal_local = getattr(worker, "seal_local", None)
            if not callable(seal_local):
                failures.append("observer worker %s has no local seal route" % run_key)
                continue
            try:
                seal_local(local_error)
            except BaseException as caught:
                failures.append(
                    "observer worker %s local seal failed: %s" % (run_key, _exception_text(caught))
                )
        return tuple(failures)

    def _refuse_lost_observer_world(self) -> None:
        reason = getattr(self, "_observer_world_collective_lost", None)
        if reason is not None:
            raise RuntimeError(
                "post-commit MPI_COMM_WORLD is sealed after collective proof loss: %s" % reason
            )

    def require_observer_world_available(self) -> None:
        """Refuse reuse of a RuntimeInstance whose observer control world lost proof."""

        self._refuse_lost_observer_world()

    def failed_run_effect_fence(self) -> str:
        """Authenticate publisher state whose mutation makes a run identity non-reusable."""

        def encoded(value: Any) -> str:
            collective = getattr(value, "to_collective_data", None)
            if callable(collective):
                value = collective()
            else:
                data = getattr(value, "to_data", None)
                if callable(data):
                    value = data()
            return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

        pending = getattr(self, "_pending", {})
        pending_baselines = getattr(self, "_pending_baselines", {})
        diagnostics = getattr(self, "_diagnostics", {})
        baselines = getattr(self, "_baselines", {})
        observer_reports = getattr(self, "_observer_reports", {})
        observer_pending_reports = getattr(self, "_observer_pending_reports", {})
        observer_run_authorities = getattr(self, "_observer_report_run_authorities", {})
        observer_failures = getattr(self, "_observer_pending_failures", {})
        payload = {
            "pending": [
                [key, [encoded(value) for value in pending[key]]] for key in sorted(pending)
            ],
            "pending_baselines": [
                [
                    key,
                    [
                        [name, float(value).hex()]
                        for name, value in sorted(pending_baselines[key].items())
                    ],
                ]
                for key in sorted(pending_baselines)
            ],
            "diagnostics": [[key, encoded(diagnostics[key])] for key in sorted(diagnostics)],
            "baselines": [[key, float(baselines[key]).hex()] for key in sorted(baselines)],
            "observer_journals": [
                list(key) for key in sorted(getattr(self, "_observer_journals", {}))
            ],
            "observer_preflight_sessions": sorted(
                getattr(self, "_observer_preflight_sessions", {})
            ),
            "observer_reports": [
                [key, encoded(observer_reports[key])] for key in sorted(observer_reports)
            ],
            "observer_pending_reports": [
                [list(key), [encoded(report) for report in observer_pending_reports[key]]]
                for key in sorted(observer_pending_reports)
            ],
            "observer_report_run_authorities": [
                [list(key), sorted(identity.token for identity in observer_run_authorities[key])]
                for key in sorted(observer_run_authorities)
            ],
            "observer_failures": [
                [list(key), list(observer_failures[key])] for key in sorted(observer_failures)
            ],
            "observer_diagnostics": list(getattr(self, "_observer_diagnostics", ())),
            "builtin_catalyst_started": bool(getattr(self, "_builtin_catalyst_run_started", False)),
        }
        return make_identity("failed-run-consumer-fence", payload).token

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
        self,
        consumer_id: str,
        run_identity: Identity,
        error: BaseException,
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
        root = configured.root / manifest.identity.hexdigest / ("rank-%06d" % self._rank)
        current = DurableJournal(root, sync=configured.sync, recover=configured.recover)
        target = Path(manifest.target_uri)
        if self._owner._output_root is not None:
            target = Path(self._owner._output_root) / target
        current.bind_delivery_authority(
            {
                "schema_version": 1,
                "consumer_id": manifest.qualified_id,
                "manifest_identity": manifest.identity.token,
                "target_uri": manifest.target_uri,
                "resolved_target": target.expanduser().resolve().as_posix(),
            }
        )
        self._observer_journals[key] = current
        return current

    @staticmethod
    def _journal_event(record: Any) -> str:
        frame = getattr(record, "frame", None)
        if type(frame) is not ObserverFrame:
            raise TypeError("durable journal record contains no exact ObserverFrame")
        request = frame.request.to_data()
        request.pop("rank")
        return make_identity(
            "durable-observer-event",
            {
                "run_identity": frame.snapshot.provenance.run_identity.to_data(),
                "clock": frame.snapshot.clock.to_data(),
                "request": request,
            },
        ).token

    def _inspect_observer_journal(
        self,
        manifest: Any,
        journal: Any,
    ) -> tuple[tuple[Any, ...], tuple[tuple[str, ...], ...]]:
        """Authenticate replay order/state before any observer session is initialized."""

        worker_mpi = manifest.parallel_mode in (ParallelMode.PER_RANK, ParallelMode.COLLECTIVE)
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
                            "durable observer journal contains duplicate committed events"
                        )
                    seen.add(event)
                    local_events.append({"event": event, "state": record.state})
            except BaseException as error:
                local_error = _exception_text(error)
                records = ()
                local_events = []
            try:
                rows = allgather_value(
                    self._communicator,
                    {
                        "rank": self._rank,
                        "events": local_events,
                        "error": local_error,
                    },
                )
            except BaseException as error:
                lost = _ObserverCollectiveLost(
                    "durable MPI observer replay lost its WORLD inspection proof: %s"
                    % _exception_text(error)
                )
                self.seal_observer_collective_loss(lost)
                raise lost from error
            if len(rows) != self._size or any(
                not isinstance(row, Mapping)
                or set(row) != {"rank", "events", "error"}
                or row["rank"] != owner
                or not isinstance(row["events"], (tuple, list))
                or (row["error"] is not None and not isinstance(row["error"], str))
                for owner, row in enumerate(rows)
            ):
                lost = _ObserverCollectiveLost(
                    "durable MPI observer replay returned malformed WORLD rank evidence"
                )
                self.seal_observer_collective_loss(lost)
                raise lost
            failures = [
                "rank %d: %s" % (owner, row["error"])
                for owner, row in enumerate(rows)
                if row["error"] is not None
            ]
            if failures:
                raise RuntimeError(
                    "durable MPI observer journal inspection failed collectively: "
                    + "; ".join(failures)
                )
            sequences = []
            rank_states = []
            for row in rows:
                sequence = []
                states = []
                for item in row["events"]:
                    if (
                        not isinstance(item, Mapping)
                        or set(item) != {"event", "state"}
                        or not isinstance(item["event"], str)
                        or item["state"] not in {"pending", "delivered"}
                        or item["event"] in sequence
                    ):
                        raise RuntimeError(
                            "durable MPI observer replay contains malformed event evidence"
                        )
                    sequence.append(item["event"])
                    states.append(item["state"])
                sequences.append(tuple(sequence))
                rank_states.append(tuple(states))
            if any(sequence != sequences[0] for sequence in sequences[1:]):
                raise RuntimeError(
                    "durable MPI observer journals disagree in temporal event order after a "
                    "crash; the handoff is not atomic with the numerical checkpoint"
                )
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

        worker_mpi = manifest.parallel_mode in (ParallelMode.PER_RANK, ParallelMode.COLLECTIVE)
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
                delivery_error = None
                try:
                    observer_queue.flush()
                except BaseException as error:
                    delivery_error = _exception_text(error)
                _post_commit_root_consensus(
                    self._communicator,
                    rank=self._rank,
                    size=self._size,
                    error=delivery_error,
                    phase="durable replay delivery %d" % index,
                )
            return
        for record in records:
            observer_queue.submit(record.frame, journal=journal, journal_record=record)

    def _open_observer_session(
        self,
        manifest: Any,
        run_identity: Identity,
        lane: Any,
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
                    None if self._owner._output_root is None else str(self._owner._output_root)
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
        defer_initialize: bool = False,
    ) -> PostCommitObserverQueue:
        key = self._observer_key(manifest.qualified_id, run_identity)
        if key in self._observer_queues:
            current = self._observer_queues[key]
            if current is None:
                raise RuntimeError("post-commit queue construction is already reserved")
            return current
        operation_data = manifest.operation_data
        if operation_data is None:
            raise RuntimeError("post-commit consumer manifest lost its operation authority")
        lane = self._observer_lanes.get(key)
        if session is None:
            session = self._open_observer_session(manifest, run_identity, lane)
        observer_run = ObserverRun(
            run_identity,
            {
                "consumer_id": manifest.qualified_id,
                "manifest_identity": manifest.identity.token,
                "bind_identity": self._owner.bind_identity.token,
            },
            recovery_run_identities,
        )
        accepted_runs = frozenset(observer_run.accepted_run_identities)
        report_authorities = getattr(self, "_observer_report_run_authorities", None)
        if report_authorities is None:
            report_authorities = {}
            self._observer_report_run_authorities = report_authorities
        retained_runs = report_authorities.get(key)
        if retained_runs is not None and retained_runs != accepted_runs:
            raise RuntimeError("observer queue report run authority changed during construction")
        report_authorities[key] = accepted_runs
        self._observer_queues[key] = None
        try:
            current = PostCommitObserverQueue(
                session,
                observer_run,
                consumer_id=manifest.qualified_id,
                capacity=operation_data["queue_capacity"],
                max_attempts=operation_data["max_attempts"],
                thread_name="pops-live-%s" % manifest.identity.hexdigest[:12],
                worker_communicator=lane,
                shared_worker=self._observer_worker(run_identity),
                defer_initialize=defer_initialize,
            )
        except BaseException:
            if self._observer_queues.get(key) is None:
                self._observer_queues.pop(key, None)
            if self._observer_queues.get(key) is None:
                report_authorities.pop(key, None)
            raise
        self._observer_queues[key] = current
        return current

    def _observer_worker(self, run_identity: Identity) -> PostCommitObserverWorker:
        self._observer_key("worker", run_identity)
        run_key = run_identity.token
        if run_key in self._observer_workers:
            current = self._observer_workers[run_key]
            if current is None:
                raise RuntimeError("post-commit worker construction is already reserved")
            return current
        self._observer_workers[run_key] = None
        try:
            current = PostCommitObserverWorker(
                thread_name="pops-post-commit-%s" % run_identity.hexdigest[:12],
                run_identity=run_identity,
            )
        except BaseException:
            if self._observer_workers.get(run_key) is None:
                self._observer_workers.pop(run_key, None)
            raise
        self._observer_workers[run_key] = current
        return current

    def _drain_post_commit_before_hdf5(self) -> None:
        """Exclude process-global observer-library calls from synchronous HDF5 publication."""

        for key in sorted(self._observer_queues):
            observer_queue = self._observer_queues[key]
            if observer_queue is None:
                raise RuntimeError("post-commit queue construction remained reserved")
            observer_queue.flush()

    def begin_post_commit_consumers(self, run_identity: Identity) -> None:
        """Initialize every active post-commit session before the first consumer/step.

        A provider may allocate run-scoped state from ``ObserverRun``.  Deferring that work until
        the first scheduled frame would allow a dependency or initialization failure after the
        numerical clock had already advanced.  ROOT ranks always exchange one status envelope
        before any rank exposes a local failure.
        """

        self._refuse_lost_observer_world()
        self._observer_key("run-begin", run_identity)
        if run_identity.token in self._closed_observer_runs:
            raise RuntimeError("post-commit consumers cannot reopen an already closed run")
        if run_identity.token in self._observer_run_phases:
            raise RuntimeError("post-commit consumers already own lifecycle state for this run")
        self._observer_run_phases[run_identity.token] = "opening"
        if self._root_output_consumers:
            if run_identity.token in self._root_output_lanes:
                raise RuntimeError(
                    "the ROOT scientific-output MPI lane is already active for this run"
                )
            if self._communicator is None:
                raise RuntimeError(
                    "ROOT scientific output lost its authenticated execution communicator"
                )
            lane_identity = "scientific-output/root/%s" % run_identity.token
            self._root_output_lanes[run_identity.token] = None
            lane_error = None
            try:
                lane = self._communicator.duplicate_observer_lane(lane_identity)
            except BaseException as error:
                lane_error = _exception_text(error)
            else:
                self._root_output_lanes[run_identity.token] = lane
            if self._size > 1:
                lane_rows = self._collective_close_rows(
                    "ROOT scientific-output lane construction",
                    {
                        "rank": self._rank,
                        "error": lane_error,
                        "present": self._root_output_lanes[run_identity.token] is not None,
                    },
                )
                malformed = any(
                    (row["error"] is not None and not isinstance(row["error"], str))
                    or type(row["present"]) is not bool
                    or (row["error"] is None) is not row["present"]
                    for row in lane_rows
                )
                if malformed:
                    raise _ObserverCollectiveRejected(
                        "ROOT scientific-output lane construction returned malformed evidence"
                    )
                failures = tuple(
                    "rank %d: %s" % (row["rank"], row["error"])
                    for row in lane_rows
                    if row["error"] is not None
                )
                if failures:
                    if not any(row["present"] is True for row in lane_rows):
                        self._root_output_lanes.pop(run_identity.token, None)
                    raise _ObserverCollectiveRejected(
                        "ROOT scientific-output lane construction failed: " + "; ".join(failures)
                    )
            elif lane_error is not None:
                self._root_output_lanes.pop(run_identity.token, None)
                raise RuntimeError(
                    "ROOT scientific-output lane construction failed: %s" % lane_error
                )
        if self._builtin_catalyst_consumers:
            if self._builtin_catalyst_run_started:
                raise RuntimeError(
                    "the built-in Catalyst lifecycle permits one run in this OS process; launch "
                    "a new process for another Catalyst simulation run"
                )
            self._builtin_catalyst_run_started = True
        manifests = self._monitor_manifests()
        for manifest in manifests:
            local_error = None
            journal = None
            replay_records: tuple[Any, ...] = ()
            replay_states: tuple[tuple[str, ...], ...] = ((),)
            worker_mpi = manifest.parallel_mode in (ParallelMode.PER_RANK, ParallelMode.COLLECTIVE)
            key = self._observer_key(manifest.qualified_id, run_identity)
            if worker_mpi:
                self._observer_lanes[key] = None
                try:
                    lane_identity = "post-commit/%s/%s" % (
                        manifest.identity.token,
                        run_identity.token,
                    )
                    lane = self._communicator.duplicate_observer_lane(lane_identity)
                    self._observer_lanes[key] = lane
                except BaseException as error:
                    if self._observer_lanes.get(key) is None:
                        self._observer_lanes.pop(key, None)
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
                _post_commit_root_consensus(
                    self._communicator,
                    rank=self._rank,
                    size=self._size,
                    error=local_error,
                    phase="journal/lane construction",
                )
            elif local_error is not None:
                raise RuntimeError("post-commit journal construction failed: %s" % local_error)

            local_error = None
            if active and journal is not None:
                try:
                    replay_records, replay_states = self._inspect_observer_journal(
                        manifest, journal
                    )
                except _ObserverCollectiveLost:
                    raise
                except BaseException as error:
                    local_error = _exception_text(error)
            if manifest.parallel_mode is not ParallelMode.SERIAL:
                _post_commit_root_consensus(
                    self._communicator,
                    rank=self._rank,
                    size=self._size,
                    error=local_error,
                    phase="durable journal inspection",
                )
            elif local_error is not None:
                raise RuntimeError("post-commit journal inspection failed: %s" % local_error)

            if worker_mpi:
                local_error = None
                try:
                    self._observer_worker(run_identity)
                except BaseException as error:
                    local_error = _exception_text(error)
                _post_commit_root_consensus(
                    self._communicator,
                    rank=self._rank,
                    size=self._size,
                    error=local_error,
                    phase="post-commit worker construction",
                )

            local_error = None
            if active:
                self._observer_pending_sessions[key] = None
                try:
                    session = self._open_observer_session(
                        manifest, run_identity, self._observer_lanes.get(key)
                    )
                    provider_id = _observer_provider_id(manifest.operation_data)
                    pending_session = _PendingObserverSession(
                        run_identity,
                        manifest.qualified_id,
                        provider_id,
                        worker_mpi,
                        session,
                    )
                    self._observer_pending_sessions[key] = pending_session
                    if not pending_session.authenticated:
                        local_error = pending_session.authentication_error
                except BaseException as error:
                    if self._observer_pending_sessions.get(key) is None:
                        self._observer_pending_sessions.pop(key, None)
                    local_error = _exception_text(error)
            # The run worker already exists on every MPI rank, so any retained session can later
            # execute its fail-closed abort on the same owner thread.
            if manifest.parallel_mode is not ParallelMode.SERIAL:
                _post_commit_root_consensus(
                    self._communicator,
                    rank=self._rank,
                    size=self._size,
                    error=local_error,
                    phase="session construction/cleanup-authority registration",
                )
            elif local_error is not None:
                raise RuntimeError("post-commit session construction failed: %s" % local_error)

            local_error = None
            if manifest.qualified_id in self._builtin_catalyst_consumers:
                try:
                    _reserve_builtin_catalyst_process_lifecycle()
                except BaseException as error:
                    local_error = _exception_text(error)
            reservation_error = None
            try:
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
                        "Catalyst process lifecycle reservation failed: %s" % local_error
                    )
            except BaseException as error:
                reservation_error = error
            if reservation_error is not None:
                # Failed-run close owns the one authenticated abort route.  Aborting here would
                # let successful ranks replay a non-idempotent collective when another rank fails
                # and the retained pending session is retried later.
                raise reservation_error

            recovery_run_identities = tuple(
                sorted(
                    {
                        record.frame.snapshot.provenance.run_identity
                        for record in replay_records
                        if record.frame.snapshot.provenance.run_identity != run_identity
                    },
                    key=lambda item: item.token,
                )
            )
            local_error = None
            if active:
                try:
                    pending_session = self._observer_pending_sessions.get(key)
                    if pending_session is None:
                        raise RuntimeError(
                            "post-commit queue construction lost its pending session authority"
                        )
                    if not pending_session.authenticated:
                        raise RuntimeError(
                            "post-commit queue construction refused an unauthenticated session"
                        )
                    observer_queue = self._observer_queue(
                        manifest,
                        run_identity,
                        session=pending_session,
                        recovery_run_identities=recovery_run_identities,
                        defer_initialize=worker_mpi,
                    )
                    self._observer_pending_sessions.pop(key, None)
                except BaseException as error:
                    local_error = _exception_text(error)
            if manifest.parallel_mode is not ParallelMode.SERIAL:
                _post_commit_root_consensus(
                    self._communicator,
                    rank=self._rank,
                    size=self._size,
                    error=local_error,
                    phase="observer queue construction",
                )
            elif local_error is not None:
                raise RuntimeError("post-commit session initialization failed: %s" % local_error)

            if worker_mpi:
                observer_queue = self._observer_queues.get(key)
                local_error = None
                try:
                    if observer_queue is None:
                        raise RuntimeError("MPI observer initialization lost its constructed queue")
                    observer_queue.prepare_initialize()
                except BaseException as error:
                    local_error = _exception_text(error)
                admission_error = None
                try:
                    _post_commit_root_consensus(
                        self._communicator,
                        rank=self._rank,
                        size=self._size,
                        error=local_error,
                        phase="observer initialization enqueue",
                    )
                except BaseException as error:
                    admission_error = error
                if admission_error is not None:
                    if observer_queue is not None:
                        observer_queue.cancel_initialize(admission_error)
                    raise admission_error
                if observer_queue is None:  # pragma: no cover - enqueue consensus proved it
                    raise RuntimeError("MPI observer initialization lost its queue")

                local_error = None
                try:
                    observer_queue.arm_initialize()
                    observer_queue.complete_initialize()
                except BaseException as error:
                    local_error = _exception_text(error)
                _post_commit_root_consensus(
                    self._communicator,
                    rank=self._rank,
                    size=self._size,
                    error=local_error,
                    phase="observer initialization completion",
                )

            local_error = None
            if active and journal is not None:
                try:
                    observer_queue = self._observer_queues.get(key)
                    if observer_queue is None:
                        raise RuntimeError("durable replay lost its initialized observer queue")
                    self._replay_observer_journal(
                        manifest,
                        observer_queue,
                        journal,
                        replay_records,
                        replay_states,
                    )
                except _ObserverCollectiveLost:
                    raise
                except BaseException as error:
                    local_error = _exception_text(error)
            if manifest.parallel_mode is not ParallelMode.SERIAL:
                _post_commit_root_consensus(
                    self._communicator,
                    rank=self._rank,
                    size=self._size,
                    error=local_error,
                    phase="durable session replay",
                )
            elif local_error is not None:
                raise RuntimeError("post-commit durable replay failed: %s" % local_error)
        self._observer_run_phases[run_identity.token] = "open"

    def _submit_live_visualization(
        self,
        effect: AcceptedSideEffect,
        frame: _DetachedObserverFrame | None,
        journal: Any = None,
        journal_record: Any = None,
        preexisting_committed: bool = False,
    ) -> None:
        """Commit and arm one post-commit job only after rank-identical main-thread consensus."""
        self._refuse_lost_observer_world()
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
            run_identity = None if raw_frame is None else raw_frame.snapshot.provenance.run_identity
        active = self._rank == 0 or manifest.parallel_mode in (
            ParallelMode.PER_RANK,
            ParallelMode.COLLECTIVE,
        )
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
                    committed_record is not None and committed_record.state == "delivered"
                )
                if not preexisting_committed and not already_delivered:
                    submission = self._observer_queue(manifest, run_identity)._prepare_detached(
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
            if isinstance(consensus_error, _ObserverCollectiveLost):
                self.seal_observer_collective_loss(consensus_error)
                raise consensus_error
            if active and type(run_identity) is Identity and run_identity.domain == "run":
                self._record_observer_failure(manifest.qualified_id, run_identity, consensus_error)
            return None
        if type(run_identity) is not Identity or run_identity.domain != "run":
            raise RuntimeError("post-commit consensus accepted no exact run identity")
        if submission is not None:
            submission.arm()
        if manifest.parallel_mode in (
            ParallelMode.PER_RANK,
            ParallelMode.COLLECTIVE,
        ):
            # A Catalyst implementation may enter MPI from its worker thread even when PoPS gives
            # it a duplicated communicator.  Do not let the next AMR/native step concurrently
            # enter solver collectives on the main thread: MPICH and third-party VTK internals do
            # not guarantee progress for that cross-library ordering.  Drain the accepted live
            # frame locally, then prove every rank has left the worker lane before any rank returns
            # to the solver.  SERIAL and gathered ROOT workers never enter MPI, so they remain
            # asynchronous with the next numerical step.
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
                if isinstance(error, _ObserverCollectiveLost):
                    self.seal_observer_collective_loss(error)
                    raise
                self._record_observer_failure(manifest.qualified_id, run_identity, error)
        return None

    def _monitor_manifests(self) -> tuple[Any, ...]:
        return tuple(
            sorted(
                (
                    row
                    for row in self._owner._consumer_graph.nodes
                    if row.kind is ConsumerKind.MONITOR
                ),
                key=lambda value: value.qualified_id,
            )
        )

    @staticmethod
    def _observer_close_state(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        return {
            "authority": getattr(value, "close_authority", None),
            "close_requested": getattr(value, "close_requested", None),
            "close_succeeded": getattr(value, "close_succeeded", None),
        }

    @staticmethod
    def _observer_lane_close_state(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        return {
            "identity": getattr(value, "identity", None),
            "active": getattr(value, "active", None),
            "closed": getattr(value, "closed", None),
        }

    @staticmethod
    def _observer_pending_session_state(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        return {
            "authority": getattr(value, "close_authority", None),
            "abort_succeeded": getattr(value, "abort_succeeded", None),
            "authenticated": getattr(value, "authenticated", None),
        }

    def _qualified_observer_lane_identity(self, local_identity: str) -> str:
        parent_identity = (
            None if self._communicator is None else getattr(self._communicator, "identity", None)
        )
        if type(parent_identity) is not str or not parent_identity:
            raise RuntimeError("observer lane lost its parent communicator identity")
        if type(local_identity) is not str or not local_identity:
            raise RuntimeError("observer lane requires a non-empty local identity")
        return "%s/%s" % (parent_identity, local_identity)

    def _collective_close_rows(
        self,
        phase: str,
        local: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        if self._size > 1 and self._communicator is None:
            lost = _ObserverCollectiveLost("%s lost its authenticated MPI communicator" % phase)
            self.seal_observer_collective_loss(lost)
            raise lost
        try:
            rows = (
                allgather_value(self._communicator, dict(local))
                if self._size > 1
                else (dict(local),)
            )
        except BaseException as error:
            lost = _ObserverCollectiveLost(
                "%s lost its MPI collective proof: %s" % (phase, _exception_text(error))
            )
            self.seal_observer_collective_loss(lost)
            raise lost from error
        keys = set(local)
        if len(rows) != self._size or any(
            not isinstance(row, Mapping) or set(row) != keys or row.get("rank") != owner
            for owner, row in enumerate(rows)
        ):
            lost = _ObserverCollectiveLost("%s returned a malformed MPI envelope" % phase)
            self.seal_observer_collective_loss(lost)
            raise lost
        return tuple(rows)

    def _seal_poisoned_observer_worker(
        self,
        run_identity: Identity,
        message: str,
    ) -> str:
        """Stop one poisoned run worker locally, then prove that stop on WORLD."""

        local_error = _ObserverWorkerLaneLost(message)
        failures: list[str] = []
        for key in sorted(getattr(self, "_observer_queues", {})):
            if len(key) != 2 or key[1] != run_identity.token:
                continue
            observer_queue = self._observer_queues[key]
            if observer_queue is None:
                continue
            seal_local = getattr(observer_queue, "seal_local", None)
            if not callable(seal_local):
                failures.append("observer queue %s has no local seal route" % (key[0],))
                continue
            try:
                seal_local(local_error)
            except BaseException as error:
                failures.append(
                    "observer queue %s local seal failed: %s" % (key[0], _exception_text(error))
                )
        worker = getattr(self, "_observer_workers", {}).get(run_identity.token)
        if worker is not None:
            seal_local = getattr(worker, "seal_local", None)
            if not callable(seal_local):
                failures.append("post-commit worker has no local seal route")
            else:
                try:
                    seal_local(local_error)
                except BaseException as error:
                    failures.append(
                        "post-commit worker local seal failed: %s" % _exception_text(error)
                    )
        rendered_error = "; ".join(failures) if failures else None
        worker_rows = self._collective_close_rows(
            "MPI observer poisoned worker local seal",
            {
                "rank": self._rank,
                "error": rendered_error,
                "closed": worker is None or worker.close_succeeded is True,
            },
        )
        worker_failures = tuple(
            "rank %d: %s"
            % (
                row["rank"],
                row["error"] or "local worker did not authenticate closure",
            )
            for row in worker_rows
            if row["error"] is not None or row["closed"] is not True
        )
        if worker_failures:
            message += "; local worker seal failed: " + "; ".join(worker_failures)
        return message

    def _preflight_observer_close(self, run_identity: Identity) -> bool:
        """Authenticate every retained close handle before entering its MPI lifecycle."""

        run_key = run_identity.token
        manifests = self._monitor_manifests()
        phase = getattr(self, "_observer_run_phases", {}).get(run_key)
        if phase not in {"opening", "open", "closing_opening", "closing_open", "closed"}:
            raise RuntimeError("post-commit close has no authenticated run lifecycle phase")
        local_error = None
        root_lane: dict[str, Any] | None = None
        worker: dict[str, bool] | None = None
        monitors: list[dict[str, Any]] = []
        try:
            root_lane = self._observer_lane_close_state(self._root_output_lanes.get(run_key))
            worker = self._observer_close_state(self._observer_workers.get(run_key))
            for manifest in manifests:
                key = self._observer_key(manifest.qualified_id, run_identity)
                monitors.append(
                    {
                        "consumer_id": manifest.qualified_id,
                        "mode": manifest.parallel_mode.value,
                        "session": self._observer_pending_session_state(
                            getattr(self, "_observer_pending_sessions", {}).get(key)
                        ),
                        "queue": self._observer_close_state(self._observer_queues.get(key)),
                        "lane": self._observer_lane_close_state(self._observer_lanes.get(key)),
                    }
                )
        except BaseException as error:
            local_error = _exception_text(error)
            root_lane = None
            worker = None
            monitors = []
        rows = self._collective_close_rows(
            "post-commit close preflight",
            {
                "rank": self._rank,
                "phase": phase,
                "error": local_error,
                "root_lane": root_lane,
                "worker": worker,
                "monitors": monitors,
            },
        )
        failures = tuple(
            "rank %d: %s" % (row["rank"], row["error"]) for row in rows if row["error"] is not None
        )
        if failures:
            raise RuntimeError(
                "post-commit close inventory failed collectively: %s" % "; ".join(failures)
            )
        if any(row["phase"] != phase for row in rows):
            raise RuntimeError("post-commit close refused divergent run lifecycle phases")

        def valid_close_state(value: Any) -> bool:
            return value is None or (
                isinstance(value, Mapping)
                and set(value) == {"authority", "close_requested", "close_succeeded"}
                and type(value["close_requested"]) is bool
                and type(value["close_succeeded"]) is bool
                and (not value["close_succeeded"] or value["close_requested"])
            )

        def valid_lane_state(value: Any) -> bool:
            return value is None or (
                isinstance(value, Mapping)
                and set(value) == {"identity", "active", "closed"}
                and isinstance(value["identity"], str)
                and bool(value["identity"])
                and type(value["active"]) is bool
                and type(value["closed"]) is bool
                and (value["active"], value["closed"]) in {(True, False), (False, True)}
            )

        def valid_session_state(value: Any) -> bool:
            return value is None or (
                isinstance(value, Mapping)
                and set(value) == {"authority", "abort_succeeded", "authenticated"}
                and isinstance(value["authority"], Mapping)
                and type(value["abort_succeeded"]) is bool
                and type(value["authenticated"]) is bool
            )

        if any(
            row["error"] is not None
            or not valid_lane_state(row["root_lane"])
            or not valid_close_state(row["worker"])
            or not isinstance(row["monitors"], (tuple, list))
            or len(row["monitors"]) != len(manifests)
            for row in rows
        ):
            raise RuntimeError("post-commit close preflight contains malformed handle evidence")
        worker_states = tuple(row["worker"] for row in rows)
        worker_ranks = tuple(rank for rank, state in enumerate(worker_states) if state is not None)
        if any(state is not None and state["authority"] != run_key for state in worker_states):
            raise RuntimeError("post-commit close found a worker owned by another run")
        if phase == "open" and any(
            state is not None
            and (state["close_requested"] is True or state["close_succeeded"] is True)
            for state in worker_states
        ):
            raise RuntimeError("post-commit close found a prematurely closed run worker")

        root_states = tuple(row["root_lane"] for row in rows)
        root_present = tuple(state is not None for state in root_states)
        if any(root_present) and not all(root_present):
            raise RuntimeError("post-commit close refused divergent ROOT output lane inventory")
        if self._root_output_consumers and phase == "open" and not any(root_present):
            raise RuntimeError("post-commit close lost its opened ROOT output lane")
        if all(root_present):
            expected = self._qualified_observer_lane_identity("scientific-output/root/%s" % run_key)
            root_signatures = tuple(
                (state["identity"], state["active"], state["closed"]) for state in root_states
            )
            if (
                any(signature != root_signatures[0] for signature in root_signatures[1:])
                or root_signatures[0][0] != expected
            ):
                raise RuntimeError(
                    "post-commit close refused unauthenticated ROOT output lane inventory"
                )
            if phase == "open" and root_signatures[0][1:] != (True, False):
                raise RuntimeError("post-commit close found a prematurely closed ROOT output lane")

        for index, manifest in enumerate(manifests):
            entries = tuple(row["monitors"][index] for row in rows)
            if any(
                not isinstance(entry, Mapping)
                or set(entry) != {"consumer_id", "mode", "session", "queue", "lane"}
                or entry["consumer_id"] != manifest.qualified_id
                or entry["mode"] != manifest.parallel_mode.value
                or not valid_session_state(entry["session"])
                or not valid_close_state(entry["queue"])
                or not valid_lane_state(entry["lane"])
                for entry in entries
            ):
                raise RuntimeError(
                    "post-commit close preflight contains malformed monitor evidence"
                )
            sessions = tuple(entry["session"] for entry in entries)
            queues = tuple(entry["queue"] for entry in entries)
            lanes = tuple(entry["lane"] for entry in entries)
            expected_queue_authority = {
                "run_identity": run_key,
                "consumer_id": manifest.qualified_id,
                "provider_id": _observer_provider_id(manifest.operation_data),
            }
            if any(
                state is not None and state["authority"] != expected_queue_authority
                for state in sessions
            ):
                raise RuntimeError(
                    "post-commit close found a pending session owned by another run or consumer"
                )
            if any(
                state is not None and state["authority"] != expected_queue_authority
                for state in queues
            ):
                raise RuntimeError(
                    "post-commit close found a queue owned by another run or consumer"
                )
            session_ranks = tuple(rank for rank, state in enumerate(sessions) if state is not None)
            queue_ranks = tuple(rank for rank, state in enumerate(queues) if state is not None)
            lane_ranks = tuple(rank for rank, state in enumerate(lanes) if state is not None)
            if any(
                session is not None and queue is not None
                for session, queue in zip(sessions, queues, strict=True)
            ):
                raise RuntimeError("post-commit close found duplicate session ownership")
            if phase in {"open", "closing_open"} and any(
                state is not None and state["authenticated"] is not True for state in sessions
            ):
                raise RuntimeError("post-commit close found an unauthenticated opened session")
            if phase == "open" and any(
                state is not None
                and (state["close_requested"] is True or state["close_succeeded"] is True)
                for state in queues
            ):
                raise RuntimeError("post-commit close found a prematurely closed observer queue")
            if manifest.parallel_mode is ParallelMode.SERIAL:
                if self._size != 1 or self._communicator is not None or lane_ranks:
                    raise RuntimeError("SERIAL post-commit close lost its serial topology")
                if phase == "open" and (session_ranks or queue_ranks != (0,)):
                    raise RuntimeError("post-commit close lost its opened SERIAL monitor queue")
                if phase == "closing_open" and (session_ranks or queue_ranks not in {(), (0,)}):
                    raise RuntimeError("post-commit close found a partial SERIAL monitor queue")
            elif manifest.parallel_mode is ParallelMode.ROOT:
                if session_ranks not in {(), (0,)} or queue_ranks not in {(), (0,)} or lane_ranks:
                    raise RuntimeError("post-commit close refused divergent ROOT monitor inventory")
                if phase == "open" and (session_ranks or queue_ranks != (0,)):
                    raise RuntimeError("post-commit close lost its opened ROOT monitor queue")
                if phase == "closing_open" and (session_ranks or queue_ranks not in {(), (0,)}):
                    raise RuntimeError("post-commit close found a partial ROOT monitor queue")
            else:
                all_ranks = tuple(range(self._size))
                owner_ranks = tuple(sorted((*session_ranks, *queue_ranks)))
                if lane_ranks not in {(), all_ranks}:
                    raise RuntimeError("post-commit close refused divergent MPI monitor inventory")
                if phase == "open" and (session_ranks or queue_ranks != all_ranks):
                    raise RuntimeError("post-commit close lost an opened MPI monitor queue")
                if phase == "closing_open" and (
                    session_ranks or queue_ranks not in {(), all_ranks}
                ):
                    raise RuntimeError("post-commit close found a partial opened MPI queue")
                if phase in {"opening", "closing_opening"} and owner_ranks not in {
                    (),
                    all_ranks,
                }:
                    raise RuntimeError(
                        "post-commit close found a gap in partial MPI session ownership"
                    )
                if owner_ranks and not lane_ranks:
                    raise RuntimeError(
                        "post-commit close refused an MPI queue without its worker lane"
                    )
                if owner_ranks and worker_ranks != all_ranks:
                    raise RuntimeError(
                        "post-commit close refused MPI session ownership without every run worker"
                    )
                if lane_ranks:
                    expected = self._qualified_observer_lane_identity(
                        "post-commit/%s/%s" % (manifest.identity.token, run_key)
                    )
                    lane_signatures = tuple(
                        (state["identity"], state["active"], state["closed"]) for state in lanes
                    )
                    if (
                        any(signature != lane_signatures[0] for signature in lane_signatures[1:])
                        or lane_signatures[0][0] != expected
                    ):
                        raise RuntimeError(
                            "post-commit close refused unauthenticated MPI worker lanes"
                        )
                    if phase == "open" and lane_signatures[0][1:] != (True, False):
                        raise RuntimeError(
                            "post-commit close found a prematurely closed MPI worker lane"
                        )
                    if lane_signatures[0][2] and any(
                        state is not None and not state["close_succeeded"] for state in queues
                    ):
                        raise RuntimeError(
                            "post-commit close found an open queue on an already closed MPI lane"
                        )

        mpi_monitors = any(
            manifest.parallel_mode in (ParallelMode.PER_RANK, ParallelMode.COLLECTIVE)
            for manifest in manifests
        )
        if mpi_monitors:
            required_worker_ranks = tuple(range(self._size))
        elif manifests:
            required_worker_ranks = (0,)
        else:
            required_worker_ranks = ()
        if phase == "open" and worker_ranks != required_worker_ranks:
            raise RuntimeError("post-commit close lost an opened run worker")
        if phase == "opening" and any(rank not in required_worker_ranks for rank in worker_ranks):
            raise RuntimeError("post-commit close refused an invalid partial worker inventory")

        return any(
            row["root_lane"] is not None
            or row["worker"] is not None
            or any(
                entry["session"] is not None
                or entry["queue"] is not None
                or entry["lane"] is not None
                for entry in row["monitors"]
            )
            for row in rows
        )

    def _drain_observer_manifest(
        self,
        manifest: Any,
        run_identity: Identity,
        *,
        close: bool,
    ) -> tuple[str, ...]:
        self._refuse_lost_observer_world()
        key = self._observer_key(manifest.qualified_id, run_identity)
        report_run_authorities = getattr(self, "_observer_report_run_authorities", None)
        if report_run_authorities is None:
            report_run_authorities = {}
            self._observer_report_run_authorities = report_run_authorities
        accepted_report_runs = report_run_authorities.get(key, frozenset((run_identity,)))
        if run_identity not in accepted_report_runs:
            raise RuntimeError("observer report authority excludes the active run")
        pending_reports = getattr(self, "_observer_pending_reports", None)
        if pending_reports is None:
            pending_reports = {}
            self._observer_pending_reports = pending_reports
        local_reports = pending_reports.get(key, ())

        def retain_pending_reports(values: tuple[ObserverDeliveryReport, ...]) -> None:
            retained = tuple(values)
            if any(type(report) is not ObserverDeliveryReport for report in retained):
                raise TypeError(
                    "pending observer reports require exact ObserverDeliveryReport values"
                )
            if key in pending_reports and pending_reports[key] != retained:
                raise RuntimeError(
                    "pending observer report authority differs from the closed queue reports"
                )
            pending_reports[key] = retained

        def world_lost(message: str) -> _ObserverCollectiveLost:
            lost = _ObserverCollectiveLost(message)
            self.seal_observer_collective_loss(lost)
            return lost

        local_diagnostics = list(self._observer_pending_failures.get(key, ()))
        worker_mpi = manifest.parallel_mode in (ParallelMode.PER_RANK, ParallelMode.COLLECTIVE)
        active = self._rank == 0 or worker_mpi
        phase = getattr(self, "_observer_run_phases", {}).get(run_identity.token)
        failed_open = close and phase == "closing_opening"
        pending_session = (
            getattr(self, "_observer_pending_sessions", {}).get(key) if active else None
        )
        observer_queue = self._observer_queues.get(key) if active else None
        release_lane = False
        cleanup_ready = True
        abort_retry_blocked = getattr(self, "_observer_abort_retry_blocked", None)
        if abort_retry_blocked is None:
            abort_retry_blocked = set()
            self._observer_abort_retry_blocked = abort_retry_blocked
        finalize_retry_blocked = getattr(self, "_observer_finalize_retry_blocked", None)
        if finalize_retry_blocked is None:
            finalize_retry_blocked = set()
            self._observer_finalize_retry_blocked = finalize_retry_blocked
        if close and worker_mpi:
            local_lane_lost = bool(
                (
                    observer_queue is not None
                    and getattr(observer_queue, "worker_collective_lost", False)
                )
                or (
                    pending_session is not None
                    and getattr(pending_session, "worker_collective_lost", False)
                )
            )
            lane_health_rows = self._collective_close_rows(
                "MPI observer worker lane health",
                {"rank": self._rank, "lost": local_lane_lost},
            )
            malformed_lane_health = any(type(row["lost"]) is not bool for row in lane_health_rows)
            if malformed_lane_health or any(row["lost"] is True for row in lane_health_rows):
                message = (
                    "MPI observer worker lane lost collective proof; provider cleanup and lane "
                    "reuse are sealed until process finalization"
                )
                if malformed_lane_health:
                    message += " (health evidence was malformed)"
                message = self._seal_poisoned_observer_worker(run_identity, message)
                if message not in local_diagnostics:
                    local_diagnostics.append(message)
                self._observer_pending_failures[key] = local_diagnostics
                raise _ObserverWorkerLaneLost(message)
        local_owner = pending_session is not None or observer_queue is not None
        queues_ready = True
        abort_close = failed_open
        if close and not failed_open and observer_queue is not None:
            local_abort_required = bool(getattr(observer_queue, "abort_required", False))
            if worker_mpi:
                abort_rows = self._collective_close_rows(
                    "MPI observer close route",
                    {"rank": self._rank, "abort_required": local_abort_required},
                )
                abort_close = any(row["abort_required"] is True for row in abort_rows)
            else:
                abort_close = local_abort_required
        if abort_close:
            pending_abort_attempt = None
            if key in abort_retry_blocked:
                cleanup_ready = False
                local_diagnostics.append(
                    "collective observer abort retry refused after rank-divergent completion"
                )
            elif observer_queue is not None:
                try:
                    local_reports = observer_queue.prepare_abort_close()
                except BaseException as error:
                    cleanup_ready = False
                    local_diagnostics.append(
                        "observer failed-open abort preparation failed: %s" % _exception_text(error)
                    )
            if worker_mpi:
                preparation_rows = self._collective_close_rows(
                    "MPI failed-open observer abort preparation",
                    {"rank": self._rank, "owned": local_owner, "ready": cleanup_ready},
                )
                cleanup_ready = all(row["ready"] is True for row in preparation_rows)
            abort_admission_error = None
            if cleanup_ready:
                try:
                    if pending_session is not None:
                        worker = self._observer_workers.get(run_identity.token)
                        if worker is None:
                            if worker_mpi:
                                raise RuntimeError("MPI pending observer abort lost its run worker")
                            worker = self._observer_worker(run_identity)
                        pending_abort_attempt = worker.prepare_call(pending_session.abort)
                    elif observer_queue is not None:
                        observer_queue.prepare_complete_abort_close()
                except BaseException as error:
                    abort_admission_error = _exception_text(error)
            if worker_mpi:
                admission_failure = None
                admission_collective_error = None
                try:
                    admission_rows = self._collective_close_rows(
                        "MPI failed-open observer abort enqueue",
                        {
                            "rank": self._rank,
                            "owned": local_owner,
                            "error": abort_admission_error,
                        },
                    )
                except BaseException as error:
                    admission_collective_error = error
                    admission_failure = RuntimeError(
                        "MPI observer abort enqueue consensus failed: %s" % _exception_text(error)
                    )
                else:
                    admission_failures = tuple(
                        "rank %d: %s" % (row["rank"], row["error"])
                        for row in admission_rows
                        if row["error"] is not None
                    )
                    if admission_failures:
                        admission_failure = RuntimeError(
                            "MPI observer abort enqueue failed collectively: %s"
                            % "; ".join(admission_failures)
                        )
                if admission_failure is not None:
                    if pending_abort_attempt is not None:
                        pending_abort_attempt.cancel(admission_failure)
                        try:
                            pending_abort_attempt.result()
                        except BaseException:
                            pass
                    elif observer_queue is not None:
                        observer_queue.cancel_complete_abort_close(admission_failure)
                    cleanup_ready = False
                    local_diagnostics.append(str(admission_failure))
                    if admission_collective_error is not None:
                        raise _ObserverCollectiveLost(str(admission_failure)) from (
                            admission_collective_error
                        )
            elif abort_admission_error is not None:
                cleanup_ready = False
                local_diagnostics.append(
                    "observer failed-open abort enqueue failed: %s" % abort_admission_error
                )
            completion_ready = cleanup_ready
            abort_armed = cleanup_ready and local_owner
            if cleanup_ready:
                try:
                    if pending_session is not None:
                        if pending_abort_attempt is None:
                            pending_session.abort()
                        else:
                            pending_abort_attempt.arm()
                            pending_abort_attempt.result()
                    elif observer_queue is not None:
                        observer_queue.arm_complete_abort_close()
                        local_reports = observer_queue.complete_abort_close()
                except BaseException as error:
                    completion_ready = False
                    local_diagnostics.append(
                        "observer failed-open abort failed: %s" % _exception_text(error)
                    )
            if worker_mpi:
                local_abort_lane_lost = bool(
                    (
                        observer_queue is not None
                        and getattr(observer_queue, "worker_collective_lost", False)
                    )
                    or (
                        pending_session is not None
                        and getattr(pending_session, "worker_collective_lost", False)
                    )
                )
                try:
                    completion_rows = self._collective_close_rows(
                        "MPI failed-open observer abort completion",
                        {
                            "rank": self._rank,
                            "owned": local_owner,
                            "ready": completion_ready,
                            "worker_lane_lost": local_abort_lane_lost,
                        },
                    )
                except BaseException as error:
                    completion_ready = False
                    if abort_armed:
                        abort_retry_blocked.add(key)
                    local_diagnostics.append(
                        "MPI observer abort completion consensus failed after provider entry; "
                        "retry is unsafe: %s" % _exception_text(error)
                    )
                    raise _ObserverCollectiveLost(
                        "MPI observer abort completion lost its collective proof"
                    ) from error
                else:
                    owner_success = any(
                        row["owned"] is True and row["ready"] is True for row in completion_rows
                    )
                    owner_failure = any(
                        row["owned"] is True and row["ready"] is not True for row in completion_rows
                    )
                    if abort_armed and owner_failure:
                        abort_retry_blocked.add(key)
                        local_diagnostics.append(
                            "collective observer abort failed after provider entry"
                            + (" on only a subset of MPI ranks" if owner_success else "")
                            + "; retry is unsafe"
                        )
                    malformed_abort_health = any(
                        type(row["worker_lane_lost"]) is not bool for row in completion_rows
                    )
                    if malformed_abort_health or any(
                        row["worker_lane_lost"] is True for row in completion_rows
                    ):
                        message = (
                            "MPI observer abort lost worker-lane collective proof; provider "
                            "cleanup and lane reuse are sealed until process finalization"
                        )
                        if malformed_abort_health:
                            message += " (abort health evidence was malformed)"
                        message = self._seal_poisoned_observer_worker(run_identity, message)
                        if message not in local_diagnostics:
                            local_diagnostics.append(message)
                        self._observer_pending_failures[key] = local_diagnostics
                        raise _ObserverWorkerLaneLost(message)
                    completion_ready = all(row["ready"] is True for row in completion_rows)
            cleanup_ready = completion_ready
            queues_ready = completion_ready
            if cleanup_ready:
                self._observer_pending_sessions.pop(key, None)
                if observer_queue is not None:
                    retain_pending_reports(local_reports)
                    self._observer_queues.pop(key, None)
        elif close and worker_mpi and key in finalize_retry_blocked:
            cleanup_ready = False
            local_diagnostics.append(
                "collective observer finalize retry refused after rank-divergent completion"
            )
        elif observer_queue is not None:
            try:
                if close and worker_mpi:
                    local_reports = observer_queue.prepare_close()
                else:
                    local_reports = observer_queue.close() if close else observer_queue.flush()
            except BaseException as error:
                cleanup_ready = False
                local_reports = observer_queue.reports
                local_diagnostics.append(_exception_text(error))
        if close and worker_mpi and not abort_close:
            preparation_rows = self._collective_close_rows(
                "MPI observer queue close preparation",
                {"rank": self._rank, "ready": cleanup_ready},
            )
            cleanup_ready = all(row["ready"] is True for row in preparation_rows)
            finalize_admission_error = None
            if cleanup_ready and observer_queue is not None:
                try:
                    observer_queue.prepare_complete_close()
                except BaseException as error:
                    finalize_admission_error = _exception_text(error)
            admission_failure = None
            admission_collective_error = None
            try:
                admission_rows = self._collective_close_rows(
                    "MPI observer queue finalization enqueue",
                    {
                        "rank": self._rank,
                        "owned": observer_queue is not None,
                        "error": finalize_admission_error,
                    },
                )
            except BaseException as error:
                admission_collective_error = error
                admission_failure = RuntimeError(
                    "MPI observer finalization enqueue consensus failed: %s"
                    % _exception_text(error)
                )
            else:
                admission_failures = tuple(
                    "rank %d: %s" % (row["rank"], row["error"])
                    for row in admission_rows
                    if row["error"] is not None
                )
                if admission_failures:
                    admission_failure = RuntimeError(
                        "MPI observer finalization enqueue failed collectively: %s"
                        % "; ".join(admission_failures)
                    )
            if admission_failure is not None:
                if observer_queue is not None:
                    observer_queue.cancel_complete_close(admission_failure)
                cleanup_ready = False
                local_diagnostics.append(str(admission_failure))
                if admission_collective_error is not None:
                    raise _ObserverCollectiveLost(str(admission_failure)) from (
                        admission_collective_error
                    )
            completion_ready = cleanup_ready
            finalize_armed = cleanup_ready and observer_queue is not None
            if cleanup_ready and observer_queue is not None:
                try:
                    observer_queue.arm_complete_close()
                    local_reports = observer_queue.complete_close()
                except BaseException as error:
                    completion_ready = False
                    local_reports = observer_queue.reports
                    local_diagnostics.append(_exception_text(error))
            try:
                completion_rows = self._collective_close_rows(
                    "MPI observer queue close completion",
                    {
                        "rank": self._rank,
                        "owned": observer_queue is not None,
                        "ready": completion_ready,
                    },
                )
            except BaseException as error:
                queues_ready = False
                if finalize_armed:
                    finalize_retry_blocked.add(key)
                local_diagnostics.append(
                    "MPI observer finalization completion consensus failed after provider entry; "
                    "retry is unsafe: %s" % _exception_text(error)
                )
                raise _ObserverCollectiveLost(
                    "MPI observer finalization completion lost its collective proof"
                ) from error
            else:
                owner_success = any(
                    row["owned"] is True and row["ready"] is True for row in completion_rows
                )
                owner_failure = any(
                    row["owned"] is True and row["ready"] is not True for row in completion_rows
                )
                if finalize_armed and owner_failure:
                    finalize_retry_blocked.add(key)
                    local_diagnostics.append(
                        "collective observer finalize failed after provider entry"
                        + (" on only a subset of MPI ranks" if owner_success else "")
                        + "; retry is unsafe"
                    )
                queues_ready = all(row["ready"] is True for row in completion_rows)
            if queues_ready and observer_queue is not None:
                retain_pending_reports(local_reports)
                self._observer_queues.pop(key, None)
        if close and observer_queue is not None and not worker_mpi and not abort_close:
            if observer_queue.close_succeeded is not True and not local_diagnostics:
                local_diagnostics.append(
                    "observer queue close returned without authenticated completion"
                )
        if close and worker_mpi:
            lane = self._observer_lanes.get(key)
            lane_error = None
            if queues_ready and lane is not None and lane.closed is not True:
                try:
                    lane.close_collectively()
                except BaseException as error:
                    lane_error = _exception_text(error)
                    local_diagnostics.append("worker MPI lane close failed: %s" % lane_error)
            if queues_ready:
                lane_rows = self._collective_close_rows(
                    "MPI observer lane close",
                    {
                        "rank": self._rank,
                        "error": lane_error,
                        "closed": lane is None or lane.closed is True,
                    },
                )
                release_lane = lane is not None and all(
                    row["error"] is None and row["closed"] is True for row in lane_rows
                )
        envelope = {
            "rank": self._rank,
            "reports": [report.to_collective_data() for report in local_reports],
            "diagnostics": local_diagnostics,
        }
        if manifest.parallel_mode is ParallelMode.ROOT:
            if self._communicator is None:
                raise world_lost("ROOT post-commit consumer lost its native communicator")
            try:
                rows = allgather_value(self._communicator, envelope)
            except BaseException as error:
                raise world_lost(
                    "ROOT post-commit flush lost its collective proof: %s" % _exception_text(error)
                ) from error
            if len(rows) != self._size or any(
                not isinstance(row, Mapping)
                or set(row) != {"rank", "reports", "diagnostics"}
                or row["rank"] != rank
                or not isinstance(row["reports"], (tuple, list))
                or not isinstance(row["diagnostics"], (tuple, list))
                for rank, row in enumerate(rows)
            ):
                raise world_lost("ROOT post-commit flush returned a malformed envelope")
            if any(row["reports"] or row["diagnostics"] for row in rows[1:]):
                raise RuntimeError("ROOT post-commit delivery occurred outside rank zero")
            authoritative = rows[0]
        elif worker_mpi:
            if self._communicator is None:
                raise world_lost("MPI post-commit flush lost its world communicator")
            try:
                rows = allgather_value(self._communicator, envelope)
            except BaseException as error:
                raise world_lost(
                    "MPI post-commit flush lost its collective proof: %s" % _exception_text(error)
                ) from error
            if len(rows) != self._size or any(
                not isinstance(row, Mapping)
                or set(row) != {"rank", "reports", "diagnostics"}
                or row["rank"] != owner
                or not isinstance(row["reports"], (tuple, list))
                or not isinstance(row["diagnostics"], (tuple, list))
                for owner, row in enumerate(rows)
            ):
                raise world_lost("MPI post-commit flush returned a malformed envelope")
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
            if (
                report.consumer_id != manifest.qualified_id
                or report.run_identity not in accepted_report_runs
            ):
                raise RuntimeError("post-commit report authenticates another run or session")
        for report in reports:
            self._observer_reports[report.identity.token] = report
        if close:
            pending_reports.pop(key, None)
            report_run_authorities.pop(key, None)
        if close:
            if worker_mpi:
                if release_lane:
                    self._observer_lanes.pop(key, None)
            elif observer_queue is not None and observer_queue.close_succeeded is True:
                self._observer_queues.pop(key, None)
        diagnostics = tuple(str(value) for value in authoritative["diagnostics"])
        release_diagnostics = tuple(
            "frame %s writer finalization: %s"
            % (
                report.frame_identity.token,
                report.receipt.detail["writer_finalize_error"],
            )
            for report in reports
            if report.receipt is not None
            and report.receipt.provider_id == "pops.output.async-scientific-writer.v1"
            and report.receipt.detail.get("writer_finalize_error") is not None
        )
        diagnostics += release_diagnostics
        for message in diagnostics:
            rendered = "%s [%s]: %s" % (manifest.qualified_id, run_identity.token, message)
            if rendered not in self._observer_diagnostics:
                self._observer_diagnostics.append(rendered)
        failures = list(diagnostics)
        failures.extend(
            "frame %s: %s" % (report.frame_identity.token, report.reason)
            for report in reports
            if report.status == "skipped"
        )
        self._observer_pending_failures.pop(key, None)
        if manifest.operation_data["on_failure"]["action"] == "report_only":
            return ()
        return tuple("%s: %s" % (manifest.qualified_id, message) for message in failures)

    def flush_live_visualizations(
        self,
        run_identity: Identity,
        *,
        close: bool = False,
        raise_on_failure: bool = True,
    ) -> tuple[ObserverDeliveryReport, ...]:
        """Drain every live consumer for one run, with ROOT consensus on the main thread."""
        self._refuse_lost_observer_world()
        self._observer_key("run-flush", run_identity)
        if close:
            self._closed_observer_runs.add(run_identity.token)
            if not self._preflight_observer_close(run_identity):
                self._observer_run_phases[run_identity.token] = "closed"
                return tuple(
                    report
                    for report in self.post_commit_reports
                    if report.run_identity == run_identity
                )
            current_phase = self._observer_run_phases[run_identity.token]
            if current_phase == "opening":
                self._observer_run_phases[run_identity.token] = "closing_opening"
            elif current_phase == "open":
                self._observer_run_phases[run_identity.token] = "closing_open"
            elif current_phase not in {"closing_opening", "closing_open"}:
                raise RuntimeError("post-commit close lost its lifecycle origin")
        failures = []
        manifests = self._monitor_manifests()
        for manifest in manifests:
            try:
                failures.extend(self._drain_observer_manifest(manifest, run_identity, close=close))
            except _ObserverCollectiveLost as error:
                self.seal_observer_collective_loss(error)
                raise
            except BaseException as error:
                rendered = "%s: %s" % (manifest.qualified_id, _exception_text(error))
                if rendered not in self._observer_diagnostics:
                    self._observer_diagnostics.append(rendered)
                failures.append(rendered)
        if close:
            root_lane = self._root_output_lanes.get(run_identity.token)
            root_error = None
            if root_lane is not None and root_lane.closed is not True:
                try:
                    root_lane.close_collectively()
                except BaseException as error:
                    root_error = _exception_text(error)
            root_rows = self._collective_close_rows(
                "ROOT scientific-output lane close",
                {
                    "rank": self._rank,
                    "error": root_error,
                    "closed": root_lane is None or root_lane.closed is True,
                },
            )
            root_failures = tuple(
                "ROOT scientific-output MPI lane close failed on rank %d: %s"
                % (row["rank"], row["error"])
                for row in root_rows
                if row["error"] is not None
            )
            failures.extend(root_failures)
            if root_lane is not None and all(
                row["error"] is None and row["closed"] is True for row in root_rows
            ):
                self._root_output_lanes.pop(run_identity.token, None)

            local_queues_remaining = (
                any(len(key) == 2 and key[1] == run_identity.token for key in self._observer_queues)
                or any(
                    len(key) == 2 and key[1] == run_identity.token
                    for key in getattr(self, "_observer_pending_sessions", {})
                )
                or any(
                    len(key) == 2 and key[1] == run_identity.token
                    for key in getattr(self, "_observer_pending_reports", {})
                )
            )
            queue_rows = self._collective_close_rows(
                "post-commit worker close readiness",
                {"rank": self._rank, "queues_remaining": local_queues_remaining},
            )
            worker = self._observer_workers.get(run_identity.token)
            worker_error = None
            if worker is not None and not any(
                row["queues_remaining"] is True for row in queue_rows
            ):
                try:
                    worker.close()
                except BaseException as error:
                    worker_error = _exception_text(error)
            worker_rows = self._collective_close_rows(
                "post-commit worker close",
                {
                    "rank": self._rank,
                    "error": worker_error,
                    "closed": worker is None or worker.close_succeeded is True,
                },
            )
            worker_failures = tuple(
                "post-commit worker close failed on rank %d: %s" % (row["rank"], row["error"])
                for row in worker_rows
                if row["error"] is not None
            )
            failures.extend(worker_failures)
            if worker is not None and all(
                row["error"] is None and row["closed"] is True for row in worker_rows
            ):
                self._observer_workers.pop(run_identity.token, None)
            if self._preflight_observer_close(run_identity):
                failures.append("run-scoped post-commit cleanup authority remains retained")
            else:
                self._observer_run_phases[run_identity.token] = "closed"
            for rendered in (*root_failures, *worker_failures):
                if rendered not in self._observer_diagnostics:
                    self._observer_diagnostics.append(rendered)
        if failures and raise_on_failure:
            raise RuntimeError(
                "post-commit consumer delivery failed at %s: %s"
                % ("run close" if close else "flush", "; ".join(failures))
            )
        return tuple(
            report for report in self.post_commit_reports if report.run_identity == run_identity
        )

    def flush_post_commit_consumers(
        self,
        run_identity: Identity,
        *,
        close: bool = False,
        raise_on_failure: bool = True,
    ) -> tuple[ObserverDeliveryReport, ...]:
        return self.flush_live_visualizations(
            run_identity, close=close, raise_on_failure=raise_on_failure
        )

    def close_live_visualizations(
        self,
        run_identity: Identity,
        *,
        raise_on_failure: bool = True,
    ) -> tuple[ObserverDeliveryReport, ...]:
        return self.flush_live_visualizations(
            run_identity, close=True, raise_on_failure=raise_on_failure
        )

    def close_failed_run_consumers(
        self,
        run_identity: Identity,
        *,
        release_identity: bool,
        entry_effect_fence: str | None = None,
    ) -> tuple[ObserverDeliveryReport, ...]:
        """Close a zero-progress failed run and retain its identity unless reuse is trivial.

        ``RunManifest`` identities intentionally describe execution semantics rather than an
        invocation nonce.  A run that fails before its first accepted step therefore receives the
        same identity when the caller fixes the external fault and retries from the restored entry
        boundary.  Reuse is deliberately limited to a serial RuntimeInstance with an empty
        ConsumerGraph and an unchanged publisher fence.  MPI, output and observer lifecycles stay
        sealed because opening or closing their external resources is already observable.
        """

        if type(release_identity) is not bool:
            raise TypeError("failed-run identity release decision must be an exact bool")
        run_key = run_identity.token
        already_closed = run_key in self._closed_observer_runs
        reports = self.flush_live_visualizations(
            run_identity,
            close=True,
            raise_on_failure=True,
        )
        graph = getattr(getattr(self, "_owner", None), "_consumer_graph", None)
        nodes = tuple(getattr(graph, "nodes", ()))
        current_effect_fence = None
        if (
            self._communicator is None
            and self._size == 1
            and not nodes
            and not self._root_output_consumers
            and not self._builtin_catalyst_consumers
        ):
            try:
                current_effect_fence = self.failed_run_effect_fence()
            except BaseException:
                current_effect_fence = None
        reusable = bool(
            self._communicator is None
            and self._size == 1
            and release_identity
            and entry_effect_fence is not None
            and current_effect_fence == entry_effect_fence
            and not already_closed
            and not reports
            and not nodes
            and not self._root_output_consumers
            and not self._builtin_catalyst_consumers
        )
        if reusable:
            self._closed_observer_runs.discard(run_key)
            self._observer_run_phases.pop(run_key, None)
        return reports

    def _root_output_communicator(self) -> Any:
        """Return the one active duplicated lane used by native ROOT snapshot gathers."""

        if not self._root_output_consumers:
            raise RuntimeError("the ConsumerGraph declares no ROOT snapshot consumer")
        if len(self._root_output_lanes) != 1:
            raise RuntimeError(
                "ROOT scientific output requires exactly one active run-scoped MPI lane"
            )
        lane = next(iter(self._root_output_lanes.values()))
        if lane.active is not True or lane.closed is not False:
            raise RuntimeError("ROOT scientific-output MPI lane is not active")
        return lane

    def diagnostic_restart_state(self) -> dict[str, Any]:
        """Return the complete last-accepted typed diagnostic registry."""
        baselines = dict(self._baselines)
        for pending in self._pending_baselines.values():
            for key, value in pending.items():
                previous = baselines.setdefault(key, value)
                if previous != value:
                    raise RuntimeError(
                        "staged conservation diagnostics disagree on their exact baseline"
                    )
        diagnostics = dict(self._diagnostics)
        staged_diagnostics: dict[str, DiagnosticPayload] = {}
        for pending in self._pending.values():
            for payload in pending:
                token = payload.key.identity.token
                previous = staged_diagnostics.get(token)
                if previous is not None and previous.to_data() != payload.to_data():
                    raise RuntimeError(
                        "staged diagnostics disagree on the latest payload for one exact key"
                    )
                staged_diagnostics[token] = payload
                diagnostics[token] = payload
        return {
            "schema_version": 2,
            "baselines": {key: value.hex() for key, value in sorted(baselines.items())},
            "diagnostics": [diagnostics[token].to_data() for token in sorted(diagnostics)],
        }

    @staticmethod
    def validate_diagnostic_restart_state(data: Any) -> dict[str, Any]:
        required = {"schema_version", "baselines", "diagnostics"}
        if (
            not isinstance(data, Mapping)
            or set(data) != required
            or data["schema_version"] != 2
            or not isinstance(data["baselines"], Mapping)
            or not isinstance(data["diagnostics"], list)
        ):
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
            if (
                not isinstance(row, Mapping)
                or set(row) != {"key", "value", "units", "terms"}
                or not isinstance(row["key"], Mapping)
                or set(row["key"])
                != {
                    "reference",
                    "component_manifest_identity",
                    "layout_identity",
                    "level",
                    "state_id",
                    "reduction",
                }
                or not isinstance(row["value"], str)
                or not isinstance(row["terms"], Mapping)
            ):
                raise TypeError("restart diagnostic payload has an unsupported shape")
            key_data = row["key"]
            if any(
                not isinstance(name, str) or not name or not isinstance(value, str)
                for name, value in row["terms"].items()
            ):
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
        baselines = {key: float.fromhex(value) for key, value in canonical["baselines"].items()}
        diagnostics: dict[str, DiagnosticPayload] = {}
        from pops.model import Handle

        for row in canonical["diagnostics"]:
            key_data = row["key"]
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
            diagnostics[payload.key.identity.token] = payload
        recorder = getattr(self._owner._executor, "record_program_diagnostic", None)
        if diagnostics and not callable(recorder):
            raise RuntimeError(
                "installed runtime cannot restore the accepted diagnostic inspection registry"
            )
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
        layouts = {row.handle.qualified_id: row for row in self._owner._layout_plan.layouts}
        for manifest in self._owner._consumer_graph.nodes:
            for quantity in manifest.diagnostic_quantities:
                block = _block_name(quantity.reference, component_names)
                names, roles = _conservative_metadata(self._owner, block)
                reductions = {
                    operation["reduction"] for operation in quantity.execution["operations"]
                }
                layout = layouts.get(quantity.layout_id)
                if layout is None:
                    raise KeyError("diagnostic selected unknown layout %s" % quantity.layout_id)
                engine = self._owner._executor_for_block(block)
                if "accepted_balance" in reductions and reductions != {"accepted_balance"}:
                    raise ValueError(
                        "accepted balance evidence cannot be mixed with field reductions"
                    )
                if reductions == {"accepted_balance"}:
                    if len(quantity.execution["operations"]) != 1:
                        raise ValueError(
                            "accepted balance requires exactly one native evidence route"
                        )
                    (operation,) = quantity.execution["operations"]
                    automatic_terms = tuple(operation.get("automatic_terms", ()))
                    if automatic_terms:
                        if not callable(getattr(engine, "_selected_accepted_balance_terms", None)):
                            raise NotImplementedError(
                                "automatic balance terms require native "
                                "_selected_accepted_balance_terms(...)"
                            )
                        component = operation["balance_component"]
                        if component >= len(names):
                            raise ValueError(
                                "automatic balance component %d is outside block %r width %d"
                                % (component, block, len(names))
                            )
                        if quantity.execution["role"] is not None:
                            role_component, _ = self._diagnostic_component(
                                names, roles, quantity.execution["role"]
                            )
                            if role_component != component:
                                raise ValueError(
                                    "automatic balance role selects component %d but ledger "
                                    "declares component %d" % (role_component, component)
                                )
                        if "reflux" in automatic_terms and not layout.adaptive:
                            raise NotImplementedError(
                                "automatic reflux balance requires an adaptive hierarchy"
                            )
                        if (
                            "projection" in automatic_terms
                            and layout.geometry.cell_measure != CARTESIAN_CELL_AREA
                        ):
                            raise NotImplementedError(
                                "automatic projection balance requires exact Cartesian cell "
                                "measure support"
                            )
                    elif not callable(getattr(engine, "_accepted_balance_terms", None)):
                        raise NotImplementedError(
                            "balance diagnostic requires native _accepted_balance_terms(route)"
                        )
                    configured_levels = tuple(level.index for level in layout.levels)
                    if tuple(quantity.levels) != configured_levels:
                        raise ValueError(
                            "balance diagnostic must select the complete configured hierarchy; "
                            "a subset cannot be reconciled with the accepted Program ledger"
                        )
                    continue
                if reductions == {"step_change_l2"}:
                    if quantity.execution["role"] is not None:
                        raise ValueError("step-change norm is a whole-state diagnostic")
                    if not callable(
                        getattr(self._owner._executor_for_block(block), "_step_change_l2", None)
                    ):
                        raise NotImplementedError(
                            "step-change norm requires native _step_change_l2()"
                        )
                else:
                    self._diagnostic_component(names, roles, quantity.execution["role"])
                if layout.adaptive:
                    if not callable(getattr(engine, "composite_reduce", None)):
                        raise NotImplementedError(
                            "adaptive diagnostic levels require native "
                            "composite_reduce(block, reduction, component, levels)"
                        )
                elif quantity.levels != (0,):
                    raise ValueError("uniform diagnostic provider accepts exactly level 0")

    def _diagnostic_metric_factor(self, quantity: Any, *, composite: bool) -> float:
        if composite:
            return 1.0
        rows = [
            row
            for row in self._owner._layout_plan.layouts
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
                % geometry.cell_measure
            )
        factor = 1.0
        for length, cells in zip(geometry.lengths, geometry.cells, strict=True):
            factor *= float(length) / int(cells)
        return factor

    @staticmethod
    def _diagnostic_component(
        names: tuple[str, ...],
        roles: tuple[str, ...],
        role: Any,
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
                "available roles are %r" % (role, roles)
            )
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
                raise RuntimeError("native step-change L2 provider omitted block %r" % block)
            return float(values[block]), True
        composite = getattr(engine, "composite_reduce", None)
        if callable(composite):
            active_depth = getattr(engine, "n_levels", None)
            if not callable(active_depth):
                active_depth = getattr(engine, "nlev", None)
            if callable(active_depth):
                nlev = int(cast(Any, active_depth)())
                levels = tuple(level for level in levels if 0 <= int(level) < nlev)
                if not levels:
                    raise RuntimeError("adaptive diagnostic selected no active AMR level")
            kind = reduction + ("_all" if full_state else "")
            return float(cast(Any, composite)(block, kind, component, list(levels))), True
        if levels != (0,):
            raise ValueError("uniform diagnostic reduction accepts exactly level 0")
        native = getattr(engine, "reduce_component", None)
        if not callable(native):
            raise RuntimeError("installed runtime has no native diagnostic reduction provider")
        if full_state and reduction in {"min", "max"}:
            count = len(_conservative_names(self._owner, block))
            values = [float(cast(Any, native)(block, reduction, index)) for index in range(count)]
            return (min(values) if reduction == "min" else max(values)), False
        kind = reduction + ("_all" if full_state else "")
        return float(cast(Any, native)(block, kind, component)), False

    @staticmethod
    def _native_balance_terms(
        engine: Any,
        route: str,
        *,
        block: str,
        component: int,
        levels: tuple[int, ...],
        automatic_terms: tuple[str, ...],
    ) -> Any:
        """Read one current-attempt balance tuple from the native transaction mailbox."""
        from pops.output.diagnostics import BalanceTerms

        native_name = (
            "_selected_accepted_balance_terms" if automatic_terms else "_accepted_balance_terms"
        )
        native = getattr(engine, native_name, None)
        if not callable(native):
            raise RuntimeError("installed runtime has no accepted balance evidence provider")
        raw = (
            native(route, block, component, list(levels), list(automatic_terms))
            if automatic_terms
            else native(route)
        )
        required = {
            "storage_change",
            "outward_boundary_flux",
            "sources",
            "reflux",
            "projection",
        }
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise TypeError(
                "native accepted balance provider must return exactly storage_change, "
                "outward_boundary_flux, sources, reflux, and projection"
            )
        if any(type(raw[name]) is not float for name in required):
            raise TypeError(
                "native accepted balance provider terms must be exact floating-point scalars"
            )
        return BalanceTerms(**{name: raw[name] for name in sorted(required)})

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
            layout = next(
                (
                    row
                    for row in self._owner._layout_plan.layouts
                    if row.handle.qualified_id == quantity.layout_id
                ),
                None,
            )
            if layout is None:
                raise KeyError("diagnostic selected unknown layout %s" % quantity.layout_id)
            levels = _active_output_levels(self._owner, layout, tuple(quantity.levels))
            variables, roles = _conservative_metadata(self._owner, block)
            execution = quantity.execution
            reductions = {operation["reduction"] for operation in execution["operations"]}
            if reductions == {"accepted_balance"}:
                if "accepted_balance" in skip_reductions:
                    continue
                (operation,) = execution["operations"]
                automatic_terms = tuple(operation.get("automatic_terms", ()))
                component = operation.get("balance_component", 0)
                balance = self._native_balance_terms(
                    engine,
                    operation["balance_route"],
                    block=block,
                    component=component,
                    levels=levels,
                    automatic_terms=automatic_terms,
                )
                terms = {
                    "storage_change": balance.storage_change,
                    "outward_boundary_flux": balance.outward_boundary_flux,
                    "sources": balance.sources,
                    "reflux": balance.reflux,
                    "projection": balance.projection,
                }
                key = DiagnosticKey(
                    quantity.handle,
                    self._owner._component_manifests[block].manifest_digest,
                    self._owner.layout_identity(quantity.layout_id),
                    min(levels),
                    quantity.identity.token,
                    "discrete_balance",
                )
                values.append(DiagnosticPayload(key, balance.residual, "unspecified", terms))
                continue
            if reductions == {"step_change_l2"}:
                component, full_state = 0, True
            else:
                component, full_state = self._diagnostic_component(
                    variables, roles, execution["role"]
                )
            for operation in execution["operations"]:
                if operation["reduction"] in skip_reductions:
                    continue
                value, composite = self._native_diagnostic_reduction(
                    engine, block, operation["reduction"], component, full_state, levels
                )
                if operation["metric_weighted"]:
                    value *= self._diagnostic_metric_factor(quantity, composite=composite)
                if operation["transform"] == "sqrt":
                    if value < 0.0:
                        raise ValueError(
                            "native sum-of-squares diagnostic returned a negative value"
                        )
                    value = math.sqrt(value)
                elif operation["transform"] != "identity":
                    raise ValueError("unknown diagnostic scalar transform")
                coefficient_token = operation["coefficient"]
                if not isinstance(coefficient_token, str):
                    raise TypeError("diagnostic coefficient must be canonical float.hex() text")
                try:
                    coefficient = float.fromhex(coefficient_token)
                except (OverflowError, ValueError) as exc:
                    raise ValueError(
                        "diagnostic coefficient is not valid float.hex() text"
                    ) from exc
                if (
                    coefficient.hex() != coefficient_token
                    or not math.isfinite(coefficient)
                    or coefficient == 0.0
                ):
                    raise ValueError(
                        "diagnostic coefficient is not canonical finite nonzero binary64"
                    )
                value *= coefficient
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
                            "conservation diagnostic tolerance must be canonical float.hex() text"
                        )
                    try:
                        tolerance = float.fromhex(tolerance_token)
                    except (OverflowError, ValueError) as exc:
                        raise ValueError(
                            "conservation diagnostic tolerance is not valid float.hex() text"
                        ) from exc
                    if (
                        tolerance.hex() != tolerance_token
                        or not math.isfinite(tolerance)
                        or tolerance < 0.0
                    ):
                        raise ValueError(
                            "conservation diagnostic tolerance is not canonical finite binary64"
                        )
                    terms = {
                        "quantity": value,
                        "baseline": baseline,
                        "absolute_drift": abs(drift),
                        "tolerance": tolerance,
                    }
                    if abs(drift) > tolerance:
                        raise RuntimeError(
                            "conservation diagnostic %s drift %.17g exceeds tolerance %.17g"
                            % (quantity.handle.qualified_id, drift, tolerance)
                        )
                    baseline_updates.setdefault(baseline_key, baseline)
                    value = drift
                    reduction_name = "conservation:%s" % reduction_name
                key = DiagnosticKey(
                    quantity.handle,
                    self._owner._component_manifests[block].manifest_digest,
                    self._owner.layout_identity(quantity.layout_id),
                    min(levels),
                    quantity.identity.token,
                    reduction_name,
                )
                values.append(DiagnosticPayload(key, value, "unspecified", terms))
        return tuple(values), baseline_updates

    def _publish_diagnostics(
        self, effect: AcceptedSideEffect, values: tuple[DiagnosticPayload, ...]
    ) -> None:
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
        last_dt = None if temporal is None else temporal.controller_state.get("last_accepted_dt")
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
                    operation["reduction"] for operation in quantity.execution["operations"]
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
                manifest, skip_reductions=frozenset({"step_change_l2"})
            )
        previous = {
            value.key.identity.token: self._diagnostics.get(value.key.identity.token)
            for value in values
        }
        existed = {
            value.key.identity.token: value.key.identity.token in self._diagnostics
            for value in values
        }
        previous_baselines = {key: self._baselines.get(key) for key in baseline_updates}
        baseline_existed = {key: key in self._baselines for key in baseline_updates}

        def rollback(_effect: AcceptedSideEffect, published: tuple[DiagnosticPayload, ...]) -> None:
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
        publish_callback: Callable[[AcceptedSideEffect, tuple[DiagnosticPayload, ...]], None]
        if manifest.kind is ConsumerKind.DIAGNOSTIC:

            def publish_console(
                accepted_effect: AcceptedSideEffect, accepted_values: tuple[DiagnosticPayload, ...]
            ) -> None:
                self._publish_diagnostics(accepted_effect, accepted_values)
                self._render_console_diagnostics(
                    accepted_effect, manifest, accepted_values, unavailable=unavailable
                )

            publish_callback = publish_console
        else:
            publish_callback = self._publish_diagnostics
        return _PreparedDiagnostic(
            effect, values, publish_callback, self._discard_diagnostics, rollback
        )

    def _snapshot_for_effect(
        self,
        effect: AcceptedSideEffect,
        manifest: Any,
    ) -> tuple[OutputSnapshot, OutputRequest]:
        if not getattr(manifest, "diagnostic_quantities", ()):
            return self._owner._output_snapshot(manifest)
        token = effect.identity.token
        try:
            diagnostics = self._pending[token]
        except KeyError as error:
            raise RuntimeError(
                "scientific output diagnostics were not prepared for the accepted effect"
            ) from error
        return self._owner._output_snapshot(manifest, diagnostics)

    def _resolve_output(self, effect: AcceptedSideEffect) -> OutputPreparation:
        manifest = self._manifest(effect)
        if manifest.output_format_data["provider_id"] == "pops.output.hdf5.v1":
            self._drain_post_commit_before_hdf5()
        snapshot, request = self._snapshot_for_effect(effect, manifest)
        fmt = manifest.output_format
        format_name = manifest.output_format_data["format_name"]
        target = _target(
            effect.target.uri,
            manifest.output_format_data,
            format_name,
            snapshot,
            request,
            manifest.handle.local_id,
            self._owner._output_root,
        )
        _rank, _size, communicator = _execution_topology(self._owner)
        if effect.target.parallel_mode is ParallelMode.SERIAL:
            communicator = None
        return OutputPreparation(fmt, snapshot, request, target, communicator)

    def _prepare_live_visualization(
        self,
        effect: AcceptedSideEffect,
        manifest: Any,
    ) -> _PreparedLiveVisualization:
        snapshot, request = self._snapshot_for_effect(effect, manifest)
        frame = None
        journal = None
        journal_record = None
        local_error = None
        try:
            if request.parallel_mode is not manifest.parallel_mode:
                raise RuntimeError("live-visualization snapshot parallel mode is stale")
            active = self._rank == 0 or manifest.parallel_mode in (
                ParallelMode.PER_RANK,
                ParallelMode.COLLECTIVE,
            )
            if active:
                # The runtime snapshot owns its field arrays, but native geometry views are
                # borrowed.  Detach once at capture time, before the accepted native boundary can
                # regrid or advance, and carry private ownership evidence into the queue.
                frame = _detach_owned_observer_frame(ObserverFrame(snapshot, request))
                journal = self._observer_journal(manifest, snapshot.provenance.run_identity)
                if journal is not None:
                    journal_record = journal.prepare(_authenticated_detached_frame(frame))
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
                    % local_error
                )
        except BaseException:
            if (
                journal is not None
                and journal_record is not None
                and journal_record.state == "prepared"
            ):
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
            diagnostic = (
                self._prepare_diagnostic(effect, manifest)
                if manifest.diagnostic_quantities
                else None
            )
            try:
                live = self._prepare_live_visualization(effect, manifest)
            except BaseException:
                if diagnostic is not None:
                    diagnostic.discard()
                raise
            return live if diagnostic is None else _PreparedScientificOutput(live, diagnostic)
        if manifest.kind is ConsumerKind.SCIENTIFIC_OUTPUT:
            diagnostic = (
                self._prepare_diagnostic(effect, manifest)
                if manifest.diagnostic_quantities
                else None
            )
            try:
                installed = self._external_writers.get(manifest.qualified_id)
                if installed is not None:
                    preparation = self._resolve_output(effect)
                    if manifest.parallel_mode is ParallelMode.ROOT:
                        output = _PreparedRootExternalWriter(
                            effect,
                            preparation,
                            installed,
                            self._owner._execution_context,
                        )
                    else:
                        output = _PreparedExternalWriter(
                            effect, preparation, installed, self._owner._execution_context
                        )
                else:
                    output = self._output.prepare(effect)
            except BaseException:
                if diagnostic is not None:
                    diagnostic.discard()
                raise
            return output if diagnostic is None else _PreparedScientificOutput(output, diagnostic)
        if manifest.kind is ConsumerKind.CHECKPOINT:
            target = Path(effect.target.uri)
            if self._owner._output_root is not None:
                target = Path(self._owner._output_root) / target.name
            extension = manifest.operation_data["extension"]
            if target.suffix != extension:
                target = target.with_suffix(extension)
            return _PreparedCheckpoint(effect, self._owner, manifest.operation, target)
        raise TypeError("unsupported ConsumerKind %r" % manifest.kind)


class RuntimeOutputSnapshot:
    """Expose exact output values from one accepted RuntimeInstance snapshot."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner
        self._geometry_cache: dict[tuple[str, int, int], LevelGeometry] = {}

    def invalidate_geometry_cache(self) -> None:
        """Drop geometry snapshots when restart replaces topology under a reused epoch."""
        self._geometry_cache.clear()

    @staticmethod
    def _native_composite_integral(
        entry: Mapping[str, Any],
        key: FieldKey,
    ) -> _NativeCompositeIntegral | None:
        """Invoke one preflighted native route for its exact selected level tuple."""
        reduction_method = entry["reduction_method"]
        if reduction_method is None:
            return None
        family_identity = _field_family_identity(key)
        levels = entry["reduction_levels"]
        reducer = getattr(entry["native_engine"], reduction_method)
        return _NativeCompositeIntegral(
            family_identity, levels, float(reducer(*entry["reduction_args"]))
        )

    def _layout(self, layout_id: str) -> Any:
        rows = [
            row for row in self._owner._layout_plan.layouts if row.handle.qualified_id == layout_id
        ]
        if len(rows) != 1:
            raise KeyError("consumer selected unknown layout %s" % layout_id)
        return rows[0]

    def _geometry(self, layout: Any, level: int) -> LevelGeometry:
        engine = self._owner._executor_for_layout(layout.handle.qualified_id)
        native_engine = getattr(engine, "_s", None)
        native_geometry = getattr(native_engine, "_output_geometry_snapshot", None)
        if not callable(native_geometry):
            raise RuntimeError("scientific output requires the native output-geometry provider")
        geometry = layout.geometry
        if type(geometry) is not NormalizedGeometry:
            raise TypeError("runtime output requires an exact normalized layout geometry")
        if geometry.dimension != 2:
            raise NotImplementedError(
                "the installed scientific-output provider supports rank-2 geometry; "
                "the normalized geometry has rank %d" % geometry.dimension
            )
        base_nx, base_ny = geometry.cells
        if int(engine.nx()) != base_nx:
            raise ValueError("runtime x cell count does not match normalized layout geometry")
        if int(engine.ny()) != base_ny:
            raise ValueError("runtime y cell count does not match normalized layout geometry")
        scale = layout.levels[level].refinement
        nx, ny = base_nx * scale, base_ny * scale
        if geometry.cell_measure not in _NATIVE_CELL_MEASURES:
            raise NotImplementedError(
                "scientific output does not implement normalized cell measure %s"
                % geometry.cell_measure
            )
        epoch_provider = getattr(native_engine, "checkpoint_topology_epoch", None)
        topology_epoch = (
            int(cast(Any, epoch_provider())) if layout.adaptive and callable(epoch_provider) else 0
        )
        layout_identity = _layout_identity(layout)
        cache_key = (layout_identity.token, level, topology_epoch)
        cached = self._geometry_cache.get(cache_key)
        if cached is not None:
            return cached
        spacing = (geometry.lengths[0] / nx, geometry.lengths[1] / ny)
        next_ratio = 0
        if layout.adaptive and level + 1 < len(layout.levels):
            next_ratio = layout.levels[level + 1].refinement // layout.levels[level].refinement
        if layout.adaptive:
            native = cast(
                Mapping[str, Any],
                native_geometry(
                    level, geometry.lower, spacing, (ny, nx), next_ratio, geometry.cell_measure
                ),
            )
        else:
            native = cast(
                Mapping[str, Any],
                native_geometry(geometry.lower, spacing, (ny, nx), geometry.cell_measure),
            )
        if int(native["topology_epoch"]) != topology_epoch:
            raise RuntimeError("native output geometry changed during snapshot construction")
        native_boxes = tuple(
            cast(tuple[int, int, int, int], tuple(int(item) for item in box))
            for box in native["boxes"]
        )
        result = LevelGeometry(
            layout_identity,
            "amr" if layout.adaptive else "uniform",
            level,
            cast(tuple[float, float], geometry.lower),
            spacing,
            (ny, nx),
            native_boxes,
            native["coverage"],
            native["cell_volumes"],
            coordinate_system=geometry.coordinate_system,
            cell_measure=geometry.cell_measure,
            axis_names=cast(tuple[str, str], geometry.axis_names),
            _native_valid_cells=native["valid_cells"],
            _native_arrays=_NATIVE_GEOMETRY_ARRAYS,
        )
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
                "installed native provider lacks required %s() output view" % method_name
            )
        rows = method(*args)
        if not isinstance(rows, (tuple, list)):
            raise TypeError("%s() must return an ordered sequence of piece mappings" % method_name)
        pieces = []
        indices = set()
        required = {
            "lower",
            "upper",
            "values",
            "global_box_index",
            "owner_rank",
            "replicated",
        }
        for position, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != required:
                raise TypeError(
                    "%s()[%d] must contain exactly %s" % (method_name, position, sorted(required))
                )
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
            if mode in (ParallelMode.ROOT, ParallelMode.COLLECTIVE) and replicated and rank != 0:
                continue
            values = np.asarray(row["values"])
            if values.dtype != np.dtype(np.float64) or not values.flags.c_contiguous:
                raise TypeError(
                    "native output pieces must expose exact C-contiguous float64 values"
                )
            native_bounds = []
            for name in ("lower", "upper"):
                bound = row[name]
                if (
                    not isinstance(bound, (tuple, list))
                    or len(bound) != 2
                    or any(type(value) is not int for value in bound)
                ):
                    raise TypeError("native output %s must be an exact integer (j, i) pair" % name)
                native_bounds.append((bound[0], bound[1]))
            lower: tuple[int, int] = native_bounds[0]
            upper: tuple[int, int] = native_bounds[1]
            pieces.append(
                (
                    box_index,
                    ArrayPiece(
                        lower,
                        upper,
                        values,
                        box_index,
                        owner_rank,
                        replicated,
                    ),
                )
            )
        pieces.sort(key=lambda item: item[0])
        return tuple(piece for _, piece in pieces)

    @staticmethod
    def _validate_piece_bounds(
        pieces: tuple[ArrayPiece, ...],
        boxes: tuple[tuple[int, int, int, int], ...],
        *,
        complete: bool,
        rank: int | None = None,
    ) -> None:
        active: list[ArrayPiece] = []
        covered = 0
        for piece in sorted(pieces, key=lambda value: (value.lower, value.upper)):
            jlo, ilo = piece.lower
            jhi, ihi = piece.upper
            if piece.global_box_index >= len(boxes):
                raise ValueError("native output global_box_index lies outside geometry boxes")
            if (jlo, ilo, jhi, ihi) != boxes[piece.global_box_index]:
                raise ValueError("native output piece bounds differ from its indexed geometry box")
            if rank is not None and piece.owner_rank != rank:
                raise ValueError("rank-local native output piece has a different owner_rank")
            active = [other for other in active if other.upper[0] > jlo]
            if any(not (ihi <= other.lower[1] or other.upper[1] <= ilo) for other in active):
                raise ValueError("native output pieces overlap")
            active.append(piece)
            covered += (jhi - jlo) * (ihi - ilo)
        if complete:
            expected = sum((jhi - jlo) * (ihi - ilo) for jlo, ilo, jhi, ihi in boxes)
            if covered != expected:
                raise ValueError("native output pieces do not exactly cover valid geometry boxes")
            if {piece.global_box_index for piece in pieces} != set(range(len(boxes))):
                raise ValueError(
                    "native output pieces do not authenticate every global geometry box"
                )
            if any(piece.replicated and piece.owner_rank != 0 for piece in pieces):
                raise ValueError("complete native output uses a non-root replicated authority")

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
            "lower",
            "upper",
            "global_box_index",
            "owner_rank",
            "replicated",
        }
        for rank, row in enumerate(rows):
            pieces = row["pieces"]
            if not isinstance(pieces, (tuple, list)):
                raise TypeError("distributed output-piece metadata must be an ordered sequence")
            for piece in pieces:
                if not isinstance(piece, Mapping) or set(piece) != required_piece_keys:
                    raise TypeError("distributed output-piece metadata schema is not exact")
                index = piece["global_box_index"]
                if (
                    isinstance(index, bool)
                    or type(index) is not int
                    or index < 0
                    or index >= len(boxes)
                ):
                    raise ValueError("distributed output-piece global_box_index is invalid")
                owner = piece["owner_rank"]
                if (
                    isinstance(owner, bool)
                    or type(owner) is not int
                    or owner < 0
                    or owner >= len(rows)
                ):
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
                    raise TypeError("distributed output-piece bounds must be exact integer pairs")
                if piece["owner_rank"] != rank:
                    raise ValueError(
                        "distributed output-piece owner differs from contributing rank"
                    )
                if tuple(lower) + tuple(upper) != boxes[index]:
                    raise ValueError(
                        "distributed output-piece bounds differ from indexed geometry box"
                    )
                by_index.setdefault(index, []).append((rank, piece))
        if set(by_index) != set(range(len(boxes))):
            raise ValueError("distributed output-piece union misses global geometry boxes")
        for index, contributors in by_index.items():
            replicated = {piece["replicated"] for _, piece in contributors}
            if replicated == {False}:
                if len(contributors) != 1:
                    raise ValueError("non-replicated global geometry box has multiple contributors")
                continue
            if replicated != {True}:
                raise ValueError(
                    "global geometry box mixes replicated and non-replicated ownership"
                )
            ranks = tuple(rank for rank, _ in contributors)
            expected = tuple(range(len(rows))) if mode is ParallelMode.PER_RANK else (0,)
            if ranks != expected:
                raise ValueError(
                    "replicated global geometry box %d has an invalid contributor set" % index
                )

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
            if mode is ParallelMode.ROOT
            else method_name
        )
        try:
            native_communicator = communicator
            if mode is ParallelMode.ROOT:
                lane_provider = getattr(self._owner._publisher, "_root_output_communicator", None)
                if not callable(lane_provider):
                    raise RuntimeError("ROOT scientific output has no run-scoped MPI lane provider")
                native_communicator = lane_provider()
            local = self._local_pieces(
                native_engine,
                selected_method,
                (native_communicator, *args) if mode is ParallelMode.ROOT else args,
                mode=mode,
                rank=rank,
                require_local_owner=mode is not ParallelMode.ROOT,
            )
            if any(
                piece.values.ndim != 3 or piece.values.shape[0] != components for piece in local
            ):
                raise ValueError(
                    "native output piece component axis differs from the compiled state"
                )
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
                for owner, row in enumerate(rows)
                if row["error"] is not None
            ]
            if failures:
                raise RuntimeError("ROOT output-piece gather failed: " + "; ".join(failures))
            return local
        rows = allgather_value(
            communicator,
            {
                "rank": rank,
                "pieces": metadata,
                "error": error,
            },
        )
        if any(
            not isinstance(row, Mapping)
            or set(row) != {"rank", "pieces", "error"}
            or row["rank"] != owner
            for owner, row in enumerate(rows)
        ):
            raise RuntimeError("%s output-piece envelope schema/rank is invalid" % mode.name)
        failures = [
            "rank %d: %s" % (owner, row["error"])
            for owner, row in enumerate(rows)
            if row["error"] is not None
        ]
        if failures:
            raise RuntimeError(
                "%s output-piece preflight failed: %s" % (mode.name, "; ".join(failures))
            )
        self._validate_distributed_piece_metadata(rows, mode=mode, boxes=boxes)
        return local

    def build(
        self, manifest: Any, diagnostics: tuple[DiagnosticPayload, ...]
    ) -> tuple[OutputSnapshot, OutputRequest]:
        import numpy as np

        rank, size, communicator = _execution_topology(self._owner)
        mode = manifest.parallel_mode
        if mode is ParallelMode.SERIAL:
            if (rank, size) != (0, 1):
                raise ValueError("SERIAL output snapshot requires rank 0 / size 1")
        elif communicator is None:
            raise ValueError(
                "%s output snapshot requires a native MPI ExecutionContext" % mode.name
            )
        entries: list[dict[str, Any]] = []
        geometries: dict[tuple[str, int], LevelGeometry] = {}
        preflight_error = None
        preflight_schema: tuple[dict[str, Any], ...] = ()
        try:
            component_names = tuple(self._owner._component_manifests)
            from pops.problem.handles import FieldHandle

            for quantity in manifest.quantities:
                layout = self._layout(quantity.layout_id)
                selected = quantity.levels or tuple(row.index for row in layout.levels)
                levels = _active_output_levels(self._owner, layout, tuple(selected))
                block = _block_name(quantity.reference, component_names)
                native_cartesian_integral = (
                    layout.adaptive and layout.geometry.cell_measure == CARTESIAN_CELL_AREA
                )
                component_manifest = self._owner._component_manifests[block].manifest_digest
                for level in levels:
                    geometry = self._geometry(layout, level)
                    geometries[geometry.key] = geometry
                    if isinstance(quantity.reference, FieldHandle):
                        plan = self._owner._install_plan.artifact.plan.field_plans.get(
                            quantity.reference.local_id
                        )
                        if plan is None:
                            raise ValueError(
                                "scientific output field %r has no resolved install plan"
                                % quantity.reference.local_id
                            )
                        engine = self._owner._executor_for_layout(layout.handle.qualified_id)
                        method_name = "output_field_local_pieces"
                        args = (plan.native_options["provider_slot"], level)
                        components = (plan.operator.unknown.local_id,)
                        reduction_method = (
                            "composite_reduce_field" if native_cartesian_integral else None
                        )
                        reduction_args = (
                            plan.native_options["provider_slot"],
                            "sum",
                            0,
                            list(levels),
                        )
                    else:
                        engine = self._owner._executor_for_block(block)
                        method_name = "output_state_local_pieces"
                        args = (block, level)
                        components = _conservative_names(self._owner, block)
                        reduction_method = (
                            "composite_reduce"
                            if native_cartesian_integral and len(components) == 1
                            else None
                        )
                        reduction_args = (block, "sum", 0, list(levels))
                    native_engine = engine._s
                    reduction_levels = tuple(levels)
                    if not callable(getattr(native_engine, method_name, None)):
                        raise RuntimeError(
                            "installed native provider lacks required %s() output view"
                            % method_name
                        )
                    if reduction_method is not None and not callable(
                        getattr(native_engine, reduction_method, None)
                    ):
                        raise RuntimeError(
                            "installed native provider lacks required %s() composite reduction"
                            % reduction_method
                        )
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
                        "reduction_levels": reduction_levels,
                    }
                    entries.append(entry)
            diagnostic_schema = []
            for quantity in manifest.diagnostic_quantities:
                layout = self._layout(quantity.layout_id)
                selected = quantity.levels or tuple(row.index for row in layout.levels)
                levels = _active_output_levels(self._owner, layout, tuple(selected))
                for level in levels:
                    geometry = self._geometry(layout, level)
                    geometries[geometry.key] = geometry
                diagnostic_schema.append(
                    {
                        "quantity": quantity.identity.token,
                        "handle": quantity.handle.qualified_id,
                        "layout": quantity.layout_id,
                        "levels": list(levels),
                        "execution": thaw_data(quantity.execution),
                    }
                )
            preflight_schema = tuple(
                {
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
                }
                for entry in entries
            ) + (
                {
                    "kind": "diagnostics",
                    "quantities": diagnostic_schema,
                },
            )
        except BaseException as exc:
            preflight_error = "%s: %s" % (type(exc).__name__, exc)
        if communicator is not None:
            rows = allgather_value(
                communicator,
                {
                    "rank": rank,
                    "schema": preflight_schema,
                    "error": preflight_error,
                },
            )
            if any(
                not isinstance(row, Mapping)
                or set(row) != {"rank", "schema", "error"}
                or row["rank"] != owner
                for owner, row in enumerate(rows)
            ):
                raise RuntimeError("output snapshot preflight envelope schema/rank is invalid")
            failures = [
                "rank %d: %s" % (owner, row["error"])
                for owner, row in enumerate(rows)
                if row["error"] is not None
            ]
            if failures:
                raise RuntimeError("output snapshot local preflight failed: " + "; ".join(failures))
            if any(row["schema"] != rows[0]["schema"] for row in rows[1:]):
                raise RuntimeError("output snapshot plan/geometry differs across ranks")
        elif preflight_error is not None:
            raise RuntimeError("output snapshot local preflight failed: " + preflight_error)

        extracted: list[tuple[dict[str, Any], tuple[ArrayPiece, ...]]] = []
        for entry in entries:
            geometry = entry["geometry"]
            if communicator is None:
                pieces = self._local_pieces(
                    entry["native_engine"],
                    entry["method_name"],
                    entry["args"],
                    mode=mode,
                    rank=rank,
                )
                if any(
                    piece.values.ndim != 3 or piece.values.shape[0] != len(entry["components"])
                    for piece in pieces
                ):
                    raise ValueError(
                        "native output piece component axis differs from compiled metadata"
                    )
                self._validate_piece_bounds(pieces, geometry.boxes, complete=True, rank=rank)
            else:
                pieces = self._distributed_pieces(
                    entry["native_engine"],
                    entry["method_name"],
                    entry["args"],
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
                fields.append(
                    FieldPayload(
                        key,
                        "cell",
                        "unspecified",
                        entry["components"],
                        geometry.cell_shape,
                        pieces,
                        dtype=np.dtype(np.float64).str,
                    )
                )
                keys.append(key)
                family_identity = _field_family_identity(key)
                levels = entry["reduction_levels"]
                authority_identity = _composite_integral_authority_identity(family_identity, levels)
                if authority_identity.token not in native_integrals:
                    evidence = self._native_composite_integral(entry, key)
                    if evidence is not None:
                        native_integrals[authority_identity.token] = evidence
            selected_handles = {
                value.handle.qualified_id for value in manifest.diagnostic_quantities
            }
            selected_diagnostics = tuple(
                value
                for value in diagnostics
                if value.key.reference.qualified_id in selected_handles
                and value.key.layout_identity.token in {key[0] for key in geometries}
            )
            expected_diagnostic_count = sum(
                len(value.execution["operations"]) for value in manifest.diagnostic_quantities
            )
            if len(selected_diagnostics) != expected_diagnostic_count:
                raise RuntimeError(
                    "scientific output did not stage every exact diagnostic operation"
                )
            request = OutputRequest(
                manifest.qualified_id,
                tuple(keys),
                mode,
                rank,
                size,
                tuple(value.key for value in selected_diagnostics),
            )
            engine = self._owner._executor
            logical_clock = manifest.schedule.domain.clock
            temporal = getattr(engine, "_temporal_restart_state", None)
            if temporal is None:
                raise RuntimeError("output snapshot requires accepted qualified temporal state")
            cursor = temporal.cursor_for_clock(logical_clock)
            last_dt_hex = temporal.controller_state.get("last_accepted_dt")
            accepted_dt = 0.0 if last_dt_hex is None else float.fromhex(last_dt_hex)
            run_identity = getattr(engine, "_last_run_identity", None)
            if type(run_identity) is not Identity:
                run_identity = make_identity(
                    "run",
                    {
                        "runtime": self._owner._runtime_plan.identity.token,
                        "time": float(engine.time()).hex(),
                        "macro_step": int(engine.macro_step()),
                    },
                )
            snapshot = OutputSnapshot(
                OutputClock.at(
                    logical_clock.qualified_id,
                    engine.time(),
                    engine.macro_step(),
                    stage="accepted",
                    tick=int(cursor["tick"]),
                    level=0,
                    substep=0,
                    stage_index=0,
                    fraction=(1, 1),
                    dt=accepted_dt,
                ),
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
            rows = allgather_value(
                communicator,
                {
                    "rank": rank,
                    "canonical": canonical,
                    "error": final_error,
                },
            )
            if any(
                not isinstance(row, Mapping)
                or set(row) != {"rank", "canonical", "error"}
                or row["rank"] != owner
                for owner, row in enumerate(rows)
            ):
                raise RuntimeError("output snapshot final envelope schema/rank is invalid")
            failures = [
                "rank %d: %s" % (owner, row["error"])
                for owner, row in enumerate(rows)
                if row["error"] is not None
            ]
            if failures:
                raise RuntimeError("output snapshot finalization failed: " + "; ".join(failures))
            if any(row["canonical"] != rows[0]["canonical"] for row in rows[1:]):
                raise RuntimeError("output snapshot canonical metadata differs across ranks")
        if snapshot is None or request is None:
            raise RuntimeError("output snapshot finalization returned no exact authority")
        return snapshot, request


__all__ = ["RuntimeConsumerPublisher", "RuntimeOutputSnapshot"]
