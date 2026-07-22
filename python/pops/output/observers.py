"""Structural contracts for irreversible post-commit visualization observers.

Scientific writers publish compensatable files before the enclosing step transaction seals.  A
live observer is deliberately different: once a frame has reached Catalyst, a socket, or another
process it cannot be rolled back.  The contracts in this module therefore accept only snapshots of
an already accepted stage and describe post-commit delivery explicitly.

PoPS imports Catalyst and Conduit only when a live session opens.  :class:`Catalyst` selects the
shipped optional Python backend by default, while extensions and tests can inject another provider
with the same four-method session protocol.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Protocol, cast

from pops._frozen_data import freeze_data, thaw_data
from pops.descriptors import Descriptor
from pops.identity import Identity, canonical_bytes, make_identity
from pops.model import Handle
from pops.output.data import (
    ArrayPiece, FieldPayload, LevelGeometry, OutputRequest, OutputSnapshot,
)
from pops.time import Schedule

from .levels import AllLevels, LevelSelection


def _add_exception_note(error: BaseException, note: str) -> None:
    """Attach cleanup evidence when the running Python supports exception notes."""

    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TypeError("%s must be non-empty canonical text" % where)
    return value


def _canonical_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise TypeError("%s must be an exact dict" % where)
    canonical_bytes(value)
    # The canonical encoder has already rejected opaque values.  ``freeze_data`` gives the worker
    # an ownership boundary without retaining an author-owned nested container that can change.
    frozen = freeze_data(value, where)
    if not isinstance(frozen, Mapping):  # defensive: identity payloads are mappings here
        raise TypeError("%s did not normalize to a mapping" % where)
    return frozen


def _collective_semantic_data(value: Any, *, where: str) -> list[Any]:
    """Project canonical semantic data to the byte-free native collective language."""

    if value is None:
        return ["none"]
    if type(value) is bool:
        return ["bool", value]
    if isinstance(value, int) and not isinstance(value, bool):
        return ["int", value]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, (list, tuple)):
        return ["list", [
            _collective_semantic_data(item, where="%s[%d]" % (where, index))
            for index, item in enumerate(value)
        ]]
    if isinstance(value, Mapping):
        rows = []
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise TypeError("%s requires non-empty string keys" % where)
            rows.append([
                key,
                _collective_semantic_data(value[key], where="%s.%s" % (where, key)),
            ])
        return ["map", rows]
    raise TypeError("%s contains unsupported %s" % (where, type(value).__name__))


def _semantic_data_from_collective(node: Any, *, where: str) -> Any:
    """Inverse of :func:`_collective_semantic_data` with exact canonical checks."""

    if not isinstance(node, list) or not node or not isinstance(node[0], str):
        raise TypeError("%s has an invalid collective semantic node" % where)
    tag = node[0]
    if tag == "none" and node == ["none"]:
        return None
    if tag == "bool" and len(node) == 2 and type(node[1]) is bool:
        return node[1]
    if tag == "int" and len(node) == 2 and type(node[1]) is int:
        return node[1]
    if tag == "str" and len(node) == 2 and isinstance(node[1], str):
        return node[1]
    if tag == "bytes" and len(node) == 2 and isinstance(node[1], str):
        try:
            result = bytes.fromhex(node[1])
        except ValueError as error:
            raise ValueError("%s contains invalid hexadecimal bytes" % where) from error
        if result.hex() != node[1]:
            raise ValueError("%s contains non-canonical hexadecimal bytes" % where)
        return result
    if tag == "list" and len(node) == 2 and isinstance(node[1], list):
        return [
            _semantic_data_from_collective(item, where="%s[%d]" % (where, index))
            for index, item in enumerate(node[1])
        ]
    if tag == "map" and len(node) == 2 and isinstance(node[1], list):
        result = {}
        previous = None
        for index, row in enumerate(node[1]):
            if not isinstance(row, list) or len(row) != 2 \
                    or not isinstance(row[0], str) or not row[0]:
                raise TypeError("%s map row %d is invalid" % (where, index))
            key = row[0]
            if previous is not None and key <= previous:
                raise ValueError("%s map keys are not canonical" % where)
            previous = key
            result[key] = _semantic_data_from_collective(
                row[1], where="%s.%s" % (where, key))
        return result
    raise ValueError("%s has an unsupported collective semantic tag" % where)


@dataclass(frozen=True, slots=True)
class ObserverRun:
    """Immutable authority shared by every frame of one live-observer session."""

    run_identity: Identity
    metadata: Mapping[str, Any] = field(default_factory=dict)
    recovery_run_identities: tuple[Identity, ...] = ()
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        if type(self.run_identity) is not Identity or self.run_identity.domain != "run":
            raise TypeError("ObserverRun.run_identity must be an exact run Identity")
        metadata = _canonical_mapping(dict(self.metadata), "ObserverRun.metadata")
        recovery = tuple(self.recovery_run_identities)
        if any(type(item) is not Identity or item.domain != "run" for item in recovery):
            raise TypeError(
                "ObserverRun.recovery_run_identities must contain exact run Identities")
        if self.run_identity in recovery or len(set(recovery)) != len(recovery):
            raise ValueError(
                "ObserverRun recovery identities must be unique and exclude the active run")
        recovery = tuple(sorted(recovery, key=lambda item: item.token))
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "recovery_run_identities", recovery)
        object.__setattr__(self, "identity", make_identity("observer-run", self.to_data()))

    @property
    def accepted_run_identities(self) -> tuple[Identity, ...]:
        """Return the active run followed by authenticated crash-recovery authorities."""

        return (self.run_identity, *self.recovery_run_identities)

    def to_data(self) -> dict[str, Any]:
        return {
            "run_identity": self.run_identity.to_data(),
            "metadata": thaw_data(self.metadata),
            "recovery_run_identities": [
                item.to_data() for item in self.recovery_run_identities
            ],
        }


@dataclass(frozen=True, slots=True)
class ObserverFrame:
    """One accepted snapshot view passed to an irreversible observer.

    Runtime-native geometry may be borrowed by the source snapshot.  Call
    :func:`detach_observer_frame` (the bounded dispatcher does this automatically) before letting
    the frame outlive the accepted-step boundary.
    """

    snapshot: OutputSnapshot
    request: OutputRequest
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        if type(self.snapshot) is not OutputSnapshot or type(self.request) is not OutputRequest:
            raise TypeError("ObserverFrame requires exact OutputSnapshot and OutputRequest values")
        if self.snapshot.clock.stage != "accepted":
            raise ValueError("a live observer accepts only an accepted-stage snapshot")
        # OutputSnapshot/ArrayPiece own read-only copies of field data.  Hashing the canonical
        # projection both authenticates the callback and makes accidental frame substitution
        # visible to the completion receipt.
        object.__setattr__(self, "identity", make_identity(
            "post-commit-observer-frame", self.snapshot.to_data(self.request)))

    @property
    def physical_time(self) -> float:
        return float.fromhex(self.snapshot.clock.time_hex)

    @property
    def macro_step(self) -> int:
        return self.snapshot.clock.macro_step


def detach_observer_frame(frame: ObserverFrame) -> ObserverFrame:
    """Deep-copy every array that may outlive the accepted native snapshot boundary."""

    if type(frame) is not ObserverFrame:
        raise TypeError("detach_observer_frame requires an exact ObserverFrame")
    geometries = tuple(LevelGeometry(
        geometry.layout_identity,
        geometry.layout_kind,
        geometry.level,
        geometry.origin,
        geometry.spacing,
        geometry.cell_shape,
        geometry.boxes,
        geometry.coverage,
        geometry.cell_volumes,
        coordinate_system=geometry.coordinate_system,
        cell_measure=geometry.cell_measure,
        axis_names=geometry.axis_names,
    ) for geometry in frame.snapshot.geometries)
    fields = []
    for field_value in frame.snapshot.fields:
        pieces = tuple(ArrayPiece(
            piece.lower,
            piece.upper,
            piece.values,
            piece.global_box_index,
            piece.owner_rank,
            piece.replicated,
        ) for piece in field_value.pieces)
        fields.append(FieldPayload(
            field_value.key,
            field_value.centering,
            field_value.units,
            field_value.component_names,
            field_value.global_shape,
            pieces,
            dtype=field_value.array_dtype,
        ))
    snapshot = OutputSnapshot(
        frame.snapshot.clock,
        frame.snapshot.provenance,
        geometries,
        tuple(fields),
        dict(frame.snapshot.metadata),
        diagnostics=frame.snapshot.diagnostics,
        _native_composite_integrals=frame.snapshot._native_composite_integrals,
    )
    detached = ObserverFrame(snapshot, frame.request)
    if detached.identity != frame.identity:
        raise RuntimeError("detaching an observer frame changed its scientific identity")
    return detached


@dataclass(frozen=True, slots=True)
class ObserverReceipt:
    """Backend acknowledgement for exactly one observer frame."""

    frame_identity: Identity
    provider_id: str
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.frame_identity) is not Identity \
                or self.frame_identity.domain != "post-commit-observer-frame":
            raise TypeError("ObserverReceipt.frame_identity has the wrong identity domain")
        object.__setattr__(self, "provider_id", _text(
            self.provider_id, "ObserverReceipt.provider_id"))
        object.__setattr__(self, "detail", _canonical_mapping(
            dict(self.detail), "ObserverReceipt.detail"))

    def to_data(self) -> dict[str, Any]:
        return {
            "frame_identity": self.frame_identity.to_data(),
            "provider_id": self.provider_id,
            "detail": thaw_data(self.detail),
        }

    def to_collective_data(self) -> dict[str, Any]:
        """Return an exact byte-free projection for native structured collectives."""

        return {
            "frame_identity": self.frame_identity.token,
            "provider_id": self.provider_id,
            "detail": _collective_semantic_data(
                thaw_data(self.detail), where="ObserverReceipt.detail"),
        }

    @classmethod
    def from_data(cls, data: Any) -> ObserverReceipt:
        if not isinstance(data, Mapping) or set(data) != {
                "frame_identity", "provider_id", "detail"}:
            raise TypeError("ObserverReceipt data has an unsupported schema")
        result = cls(
            Identity.from_data(data["frame_identity"]),
            data["provider_id"],
            data["detail"],
        )
        if result.to_data() != dict(data):
            raise ValueError("ObserverReceipt data is not canonical")
        return result

    @classmethod
    def from_collective_data(cls, data: Any) -> ObserverReceipt:
        if not isinstance(data, Mapping) or set(data) != {
                "frame_identity", "provider_id", "detail"}:
            raise TypeError("ObserverReceipt collective data has an unsupported schema")
        result = cls(
            Identity.from_token(data["frame_identity"]),
            data["provider_id"],
            _semantic_data_from_collective(
                data["detail"], where="ObserverReceipt.detail"),
        )
        if result.to_collective_data() != dict(data):
            raise ValueError("ObserverReceipt collective data is not canonical")
        return result


class ObserverSession(Protocol):
    """Dedicated session owned by one post-commit delivery worker.

    ``execute`` must not retain frame array pointers after returning.  The minimal worker rejects
    MPI sessions unless the runtime supplies the exact duplicated communicator lane authenticated
    by the session.  A worker must never borrow ``MPI_COMM_WORLD`` directly.
    """

    @property
    def authority(self) -> dict[str, Any]: ...

    def initialize(self, run: ObserverRun) -> None: ...

    def execute(self, frame: ObserverFrame) -> ObserverReceipt: ...

    def finalize(self) -> None: ...

    def abort(self) -> None: ...


class ObserverProvider(Protocol):
    """Structural provider implemented by an optional visualization package."""

    def consumer_data(self) -> dict[str, Any]: ...

    def open_session(
        self, configuration: Mapping[str, Any], execution_context: Any,
    ) -> ObserverSession: ...


_SESSION_AUTHORITY_KEYS = frozenset({
    "schema_version", "provider_id", "delivery", "threading", "worker_mpi",
})


def authenticate_observer_session(session: Any) -> dict[str, Any]:
    """Authenticate a fake or optional real observer without importing its implementation."""

    for method in ("initialize", "execute", "finalize", "abort"):
        if not callable(getattr(session, method, None)):
            raise TypeError("observer session must implement %s()" % method)
    first, second = getattr(session, "authority", None), getattr(session, "authority", None)
    if type(first) is not dict or type(second) is not dict or first != second:
        raise TypeError("observer session authority must be one deterministic exact dict")
    if set(first) != _SESSION_AUTHORITY_KEYS or first["schema_version"] != 1:
        raise ValueError("observer session authority has an unsupported schema")
    _text(first["provider_id"], "observer session provider_id")
    if first["delivery"] != "post_commit":
        raise ValueError("observer session must declare irreversible post_commit delivery")
    if first["threading"] not in {"dedicated_serial", "dedicated_collective"}:
        raise ValueError(
            "observer session threading must be dedicated_serial or dedicated_collective")
    if type(first["worker_mpi"]) is not bool:
        raise TypeError("observer session worker_mpi must be an exact bool")
    if first["worker_mpi"] != (first["threading"] == "dedicated_collective"):
        raise ValueError(
            "observer session worker_mpi and threading authority disagree")
    canonical_bytes(first)
    return dict(first)


class Catalyst:
    """Immutable Catalyst declaration backed by an optional structural provider.

    The provider is expected to carry the real ParaView/Catalyst dependency.  ``pipeline`` names a
    real script whose content digest is frozen into the declaration.  ``args`` follows the
    standard ParaView-Catalyst script-argument Blueprint and is available through
    ``paraview.catalyst.get_args()``.
    """

    __pops_ir_immutable__ = True
    __slots__ = (
        "_provider", "_provider_data", "pipeline", "pipeline_sha256", "implementation",
        "search_paths", "args",
    )

    def __init__(
        self,
        *,
        pipeline: str,
        implementation: str = "paraview",
        search_paths: tuple[str, ...] = (),
        args: tuple[str, ...] = (),
        provider: ObserverProvider | None = None,
    ) -> None:
        if provider is None:
            # Importing the optional Catalyst/Conduit modules remains lazy in the provider.  The
            # public declaration therefore stays usable for compilation and inspection on hosts
            # where ParaView is not installed, and fails only if a live session is actually opened.
            from pops.output._catalyst_backend import CatalystPythonProvider

            provider = CatalystPythonProvider()
        data_method = getattr(provider, "consumer_data", None)
        open_method = getattr(provider, "open_session", None)
        if not callable(data_method) or not callable(open_method):
            raise TypeError(
                "Catalyst provider must implement consumer_data() and open_session()")
        first, second = data_method(), data_method()
        if type(first) is not dict or type(second) is not dict or first != second:
            raise TypeError("Catalyst provider consumer_data() must be deterministic")
        if first.get("schema_version") != 1 or first.get("observer_kind") != "catalyst":
            raise ValueError("Catalyst provider must declare observer_kind='catalyst' schema v1")
        _text(first.get("provider_id"), "Catalyst provider_id")
        canonical_bytes(first)
        object.__setattr__(self, "_provider", provider)
        object.__setattr__(self, "_provider_data", _canonical_mapping(
            first, "Catalyst provider data"))
        pipeline_path = Path(_text(pipeline, "Catalyst.pipeline")).expanduser().resolve()
        if not pipeline_path.is_file():
            raise FileNotFoundError("Catalyst pipeline does not exist: %s" % pipeline_path)
        object.__setattr__(self, "pipeline", pipeline_path.as_posix())
        object.__setattr__(self, "pipeline_sha256", hashlib.sha256(
            pipeline_path.read_bytes()).hexdigest())
        object.__setattr__(self, "implementation", _text(
            implementation, "Catalyst.implementation"))
        paths = tuple(search_paths)
        if any(not isinstance(value, str) for value in paths):
            raise TypeError("Catalyst.search_paths must contain path strings")
        resolved_paths = tuple(
            Path(_text(value, "Catalyst.search_paths item")).expanduser().resolve()
            for value in paths
        )
        if any(not value.is_dir() for value in resolved_paths):
            raise NotADirectoryError(
                "Catalyst.search_paths must contain existing directories")
        if len(set(resolved_paths)) != len(resolved_paths):
            raise ValueError("Catalyst.search_paths must be unique")
        object.__setattr__(self, "search_paths", tuple(
            value.as_posix() for value in resolved_paths))
        script_args = tuple(args)
        if any(not isinstance(value, str) for value in script_args):
            raise TypeError("Catalyst.args must contain strings")
        script_args = tuple(_text(value, "Catalyst.args item") for value in script_args)
        object.__setattr__(self, "args", script_args)

    def __setattr__(self, name: str, value: Any) -> None:
        del name, value
        raise AttributeError("Catalyst declarations are immutable")

    def consumer_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider_id": "pops.output.catalyst.v1",
            "observer_kind": "catalyst",
            "pipeline": self.pipeline,
            "pipeline_sha256": self.pipeline_sha256,
            "implementation": self.implementation,
            "search_paths": list(self.search_paths),
            "args": list(self.args),
            "provider": thaw_data(self._provider_data),
        }

    def open_session(self, execution_context: Any) -> ObserverSession:
        return self._open_session(self.consumer_data(), execution_context)

    def open_runtime_session(
        self, runtime_configuration: Mapping[str, Any], execution_context: Any,
    ) -> ObserverSession:
        if not isinstance(runtime_configuration, Mapping):
            raise TypeError("Catalyst runtime configuration must be a mapping")
        allowed = {"target_uri", "output_root", "consumer_id", "worker_communicator"}
        if not set(runtime_configuration).issubset(allowed):
            raise TypeError("Catalyst runtime configuration contains unsupported keys")
        configuration = self.consumer_data()
        communicator = runtime_configuration.get("worker_communicator")
        if communicator is not None:
            configuration["_pops_worker_communicator"] = communicator
        return self._open_session(configuration, execution_context)

    def _open_session(
        self, configuration: Mapping[str, Any], execution_context: Any,
    ) -> ObserverSession:
        current = self._provider.consumer_data()
        if type(current) is not dict or current != thaw_data(self._provider_data):
            raise RuntimeError("Catalyst provider changed after its declaration was authenticated")
        session = self._provider.open_session(configuration, execution_context)
        authority = authenticate_observer_session(session)
        expected_provider = current["provider_id"]
        if authority["provider_id"] != expected_provider:
            raise ValueError(
                "Catalyst session provider_id differs from its authenticated provider: "
                "%r != %r" % (authority["provider_id"], expected_provider)
            )
        return cast(ObserverSession, session)


class LiveFailurePolicy:
    """Closed post-commit delivery policy evaluated at an explicit flush boundary."""

    __pops_ir_immutable__ = True

    def to_data(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RaiseOnFlush(LiveFailurePolicy):
    """Retain delivery evidence immediately and raise when the run flushes the observer."""

    def to_data(self) -> dict[str, Any]:
        return {"action": "raise_on_flush"}


@dataclass(frozen=True, slots=True)
class ReportOnly(LiveFailurePolicy):
    """Never roll back the simulation; expose failed deliveries through runtime reports."""

    def to_data(self) -> dict[str, Any]:
        return {"action": "report_only"}


_LIVE_FAILURE_POLICIES = (RaiseOnFlush, ReportOnly)
_LIVE_FIELD_KINDS = frozenset({"state", "field", "aux"})


@dataclass(frozen=True, slots=True)
class _LiveObserverOperation:
    """Immutable manifest provider for one bounded live-visualization lane."""

    observer: Any
    parallel_mode: Any
    queue_capacity: int
    max_attempts: int
    on_failure: LiveFailurePolicy
    durability: Any = None
    _observer_data: Mapping[str, Any] = field(init=False, repr=False)
    _session_provider_id: str = field(init=False, repr=False)
    __pops_ir_immutable__: ClassVar[bool] = True

    def __post_init__(self) -> None:
        from ._consumer_contracts import ParallelMode
        from ._durable_journal import DurableJournal

        if self.durability is not None and type(self.durability) is not DurableJournal:
            raise TypeError(
                "live observer durability must be an exact DurableJournal or None")
        first, second = self.observer.consumer_data(), self.observer.consumer_data()
        if type(first) is not dict or type(second) is not dict or first != second:
            raise TypeError("live observer consumer_data() must return one deterministic dict")
        canonical_bytes(first)
        observer_kind = first.get("observer_kind")
        if observer_kind == "catalyst":
            if self.parallel_mode not in (ParallelMode.SERIAL, ParallelMode.COLLECTIVE):
                raise ValueError(
                    "Catalyst live visualization supports only SERIAL or COLLECTIVE mode")
        elif observer_kind != "async_scientific_output" \
                and self.parallel_mode is not ParallelMode.SERIAL:
            raise ValueError(
                "this live observer supports only ParallelMode.SERIAL")
        expected_provider = first.get("provider_id")
        if first.get("observer_kind") == "catalyst":
            provider = first.get("provider")
            if not isinstance(provider, Mapping):
                raise TypeError(
                    "Catalyst observer data must carry its authenticated provider mapping")
            expected_provider = provider.get("provider_id")
        expected_provider = _text(
            expected_provider, "live observer session provider_id")
        object.__setattr__(self, "_observer_data", _canonical_mapping(
            first, "live observer consumer_data"))
        object.__setattr__(self, "_session_provider_id", expected_provider)

    def _authenticate_observer(self) -> None:
        first, second = self.observer.consumer_data(), self.observer.consumer_data()
        if type(first) is not dict or type(second) is not dict or first != second \
                or first != thaw_data(self._observer_data):
            raise RuntimeError(
                "live observer changed after its declaration was authenticated")

    def consumer_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider_id": "pops.output.live-visualization.v1",
            "parallel_mode": self.parallel_mode.value,
            "queue_capacity": self.queue_capacity,
            "max_attempts": self.max_attempts,
            "on_failure": self.on_failure.to_data(),
            "durability": (
                None if self.durability is None else self.durability.to_data()),
            "observer": thaw_data(self._observer_data),
        }

    def _authenticate_session(self, session: Any) -> ObserverSession:
        authority = authenticate_observer_session(session)
        if authority["provider_id"] != self._session_provider_id:
            raise ValueError(
                "live observer session provider_id differs from its authenticated manifest: "
                "%r != %r"
                % (authority["provider_id"], self._session_provider_id)
            )
        return cast(ObserverSession, session)

    def open_session(self, execution_context: Any) -> ObserverSession:
        self._authenticate_observer()
        session = self.observer.open_session(execution_context)
        return self._authenticate_session(session)

    def open_runtime_session(
        self, runtime_configuration: Mapping[str, Any], execution_context: Any,
    ) -> ObserverSession:
        self._authenticate_observer()
        provider = getattr(self.observer, "open_runtime_session", None)
        session = (
            provider(runtime_configuration, execution_context)
            if callable(provider) else self.observer.open_session(execution_context)
        )
        return self._authenticate_session(session)

    def preflight(self, execution_context: Any) -> Any:
        self._authenticate_observer()
        provider = getattr(self.observer, "preflight", None)
        return None if not callable(provider) else provider(execution_context)

    def preopen_session(self, execution_context: Any) -> ObserverSession | None:
        self._authenticate_observer()
        provider = getattr(self.observer, "preopen_session", None)
        if callable(provider):
            session = provider(execution_context)
            if session is None:
                return None
        else:
            session = self.observer.open_session(execution_context)
        return self._authenticate_session(session)


class _AsyncScientificWriterObserver:
    """Immutable bridge from a scientific format to the post-commit observer protocol."""

    __pops_ir_immutable__ = True
    __slots__ = ("_format", "_format_data")

    def __init__(self, format_provider: Any) -> None:
        from .provider import consumer_format_data

        data = consumer_format_data(
            format_provider, where="AsyncScientificOutput.format")
        if data["provider_id"] == "pops.output.external-writer.v1":
            raise ValueError(
                "AsyncScientificOutput does not accept ExternalWriter: installed native Writers "
                "have no dedicated post-commit worker session route")
        object.__setattr__(self, "_format", format_provider)
        object.__setattr__(self, "_format_data", _canonical_mapping(
            data, "AsyncScientificOutput.format.consumer_data"))

    def __setattr__(self, name: str, value: Any) -> None:
        del name, value
        raise AttributeError("async scientific writer declarations are immutable")

    def consumer_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider_id": "pops.output.async-scientific-writer.v1",
            "observer_kind": "async_scientific_output",
            "format": thaw_data(self._format_data),
        }

    def _authenticate_format(self) -> None:
        from .provider import consumer_format_data

        current = consumer_format_data(
            self._format, where="AsyncScientificOutput.format")
        if current != thaw_data(self._format_data):
            raise RuntimeError(
                "AsyncScientificOutput format changed after its declaration was authenticated")

    def preflight(self, execution_context: Any) -> dict[str, Any]:
        self._authenticate_format()
        writer = self._format.writer()
        callback = getattr(writer, "preflight", None)
        if not callable(callback) or not callable(getattr(writer, "prepare_session", None)):
            raise TypeError(
                "AsyncScientificOutput writer must implement preflight() and prepare_session()")
        result = callback(execution_context)
        if type(result) is not dict:
            raise TypeError("AsyncScientificOutput writer preflight() must return an exact dict")
        canonical_bytes(result)
        return result

    def preopen_session(self, execution_context: Any) -> None:
        # The writer dependency and factory were authenticated by ``preflight``.  A frame session
        # additionally needs the run-time output root, so it cannot be constructed at bind time.
        del execution_context
        return None

    def open_runtime_session(
        self, configuration: Mapping[str, Any], execution_context: Any,
    ) -> ObserverSession:
        self._authenticate_format()
        return _AsyncScientificWriterSession(
            self._format,
            thaw_data(self._format_data),
            configuration,
            execution_context,
        )

    def open_session(self, execution_context: Any) -> ObserverSession:
        del execution_context
        raise RuntimeError(
            "AsyncScientificOutput requires its run-time target configuration")


class _AsyncScientificWriterSession:
    """Dedicated worker session that returns acknowledgement only for a real published file."""

    def __init__(
        self,
        format_provider: Any,
        format_data: Mapping[str, Any],
        configuration: Mapping[str, Any],
        execution_context: Any,
    ) -> None:
        expected = {"target_uri", "output_root", "consumer_id"}
        allowed = expected | {"worker_communicator"}
        if not isinstance(configuration, Mapping) or not expected.issubset(configuration) \
                or not set(configuration).issubset(allowed):
            raise TypeError("async scientific writer runtime configuration is not exact")
        target_uri = _text(configuration["target_uri"], "async output target_uri")
        output_root = configuration["output_root"]
        if output_root is not None:
            output_root = _text(output_root, "async output output_root")
        self._format = format_provider
        self._format_data = dict(format_data)
        self._target_uri = target_uri
        self._output_root = output_root
        self._consumer_id = _text(
            configuration["consumer_id"], "async output consumer_id")
        self._communicator = configuration.get("worker_communicator")
        self._execution_context = execution_context
        self._initialized = False
        self._finalized = False
        self._accepted_run_identities: frozenset[Identity] = frozenset()

    @property
    def authority(self) -> dict[str, Any]:
        from ._consumer_contracts import ParallelMode

        mode = ParallelMode(self._format_data["parallel_mode"])
        worker_mpi = mode in (ParallelMode.PER_RANK, ParallelMode.COLLECTIVE)
        return {
            "schema_version": 1,
            "provider_id": "pops.output.async-scientific-writer.v1",
            "delivery": "post_commit",
            "threading": "dedicated_collective" if worker_mpi else "dedicated_serial",
            "worker_mpi": worker_mpi,
        }

    def initialize(self, run: ObserverRun) -> None:
        if self._initialized or self._finalized:
            raise RuntimeError("async scientific writer session cannot initialize twice")
        if type(run) is not ObserverRun:
            raise TypeError("async scientific writer requires an exact ObserverRun")
        self._accepted_run_identities = frozenset(run.accepted_run_identities)
        self._initialized = True

    def _target(self, frame: ObserverFrame) -> Path:
        from pops.output._writers.common import deterministic_target

        path = Path(self._target_uri)
        if self._output_root is not None:
            path = Path(self._output_root) / path
        extension = self._format_data["extension"]
        return deterministic_target(
            path,
            self._consumer_id.rsplit("/", 1)[-1],
            frame.request,
            frame.snapshot,
            extension,
            format_data=self._format_data,
            format_name=self._format_data["format_name"],
        )

    def _phase_evidence(
        self, phase: str, error: BaseException | None, state: str,
    ) -> tuple[BaseException | None, tuple[str, ...]]:
        rendered = None if error is None else "%s: %s" % (type(error).__name__, error)
        if self._communicator is None:
            return error, (state,)
        from pops._native_collectives import allgather_value, rank, size

        rows = allgather_value(self._communicator, {
            "rank": rank(self._communicator),
            "error": rendered,
            "state": state,
        })
        if len(rows) != size(self._communicator) or any(
                not isinstance(row, dict)
                or set(row) != {"rank", "error", "state"}
                or row["rank"] != owner
                or (row["error"] is not None and not isinstance(row["error"], str))
                or not isinstance(row["state"], str)
                for owner, row in enumerate(rows)):
            return RuntimeError(
                "async scientific writer %s returned malformed rank evidence" % phase), ()
        failures = [
            "rank %d: %s" % (owner, row["error"])
            for owner, row in enumerate(rows) if row["error"] is not None
        ]
        states = tuple(row["state"] for row in rows)
        return (
            None if not failures else RuntimeError(
                "async scientific writer %s failed collectively: %s"
                % (phase, "; ".join(failures))),
            states,
        )

    def _prepare_output_session(self, frame: ObserverFrame) -> tuple[Any, Path, Any]:
        from pops.output._consumer_contracts import ParallelMode
        from pops.output._writers.common import authenticate_writer_session
        from .provider import consumer_format_data

        if not self._initialized or self._finalized:
            raise RuntimeError("async scientific writer session is not active")
        if type(frame) is not ObserverFrame:
            raise TypeError("async scientific writer requires an exact ObserverFrame")
        if frame.snapshot.provenance.run_identity not in self._accepted_run_identities:
            raise ValueError(
                "async scientific output frame is outside the active/recovery run authority")
        mode = ParallelMode(self._format_data["parallel_mode"])
        if frame.request.parallel_mode is not mode:
            raise ValueError("async scientific output frame mode differs from its format")
        if mode is ParallelMode.SERIAL:
            if (frame.request.rank, frame.request.size) != (0, 1) \
                    or self._communicator is not None:
                raise ValueError("SERIAL async scientific output has invalid topology")
        elif mode is ParallelMode.ROOT:
            if frame.request.rank != 0 or self._communicator is not None:
                raise ValueError("ROOT async scientific output runs only on detached rank zero")
        else:
            from pops._native_collectives import rank, size

            if self._communicator is None \
                    or frame.request.rank != rank(self._communicator) \
                    or frame.request.size != size(self._communicator):
                raise ValueError(
                    "distributed async scientific output requires its exact worker MPI lane")
        current = consumer_format_data(
            self._format, where="AsyncScientificOutput.format")
        if current != self._format_data:
            raise RuntimeError("async scientific output format changed during the run")
        writer = self._format.writer()
        preflight = getattr(writer, "preflight", None)
        if not callable(preflight) or type(preflight(self._execution_context)) is not dict:
            raise TypeError("async scientific writer preflight contract changed during the run")
        target = self._target(frame)
        session = writer.prepare_session(
            frame.snapshot, frame.request, target, communicator=self._communicator)
        authority = authenticate_writer_session(session)
        writer_format = getattr(writer, "format", None)
        if not isinstance(writer_format, str) or not writer_format:
            raise TypeError("async scientific writer must declare its canonical format name")
        expected_authority = {
            "format": writer_format,
            "parallel_mode": mode.value,
            "rank": frame.request.rank,
            "size": frame.request.size,
            "target": target.expanduser().resolve().as_posix(),
            "selection_identity": frame.request.publication_identity.token,
        }
        mismatches = {
            name: {"expected": expected, "actual": authority[name]}
            for name, expected in expected_authority.items()
            if authority[name] != expected
        }
        if mismatches:
            raise ValueError(
                "async writer session authority differs from its exact request: %r"
                % mismatches)
        return mode, target, session

    def execute(self, frame: ObserverFrame) -> ObserverReceipt:
        from pops.output._consumer_contracts import ParallelMode
        from pops.output._writers.common import OutputPublicationReceipt

        mode = ParallelMode(self._format_data["parallel_mode"])
        session = None
        target = None
        preparation_error = None
        try:
            mode, target, session = self._prepare_output_session(frame)
        except BaseException as error:
            preparation_error = error
        target_state = (
            "missing" if target is None
            else target.expanduser().resolve().as_posix()
        )
        failure, states = self._phase_evidence(
            "session preparation", preparation_error,
            target_state,
        )
        if failure is not None:
            raise failure
        if session is None or target is None:
            raise RuntimeError("async writer preparation lost its local session authority")
        if mode is ParallelMode.COLLECTIVE and len(set(states)) != 1:
            mismatch = RuntimeError(
                "COLLECTIVE async writer ranks resolved different target paths")
            try:
                if session.abort_prepare() is not None:
                    raise TypeError("scientific writer abort_prepare() must return None")
            except BaseException as cleanup_error:
                _add_exception_note(mismatch,
                    "async writer target-mismatch cleanup also failed: %s: %s"
                    % (type(cleanup_error).__name__, cleanup_error))
            raise mismatch

        staged = False
        stage_error = None
        try:
            result = session.stage()
            if result is not None:
                raise TypeError("scientific writer stage() must return None")
            staged = True
        except BaseException as error:
            stage_error = error
        failure, states = self._phase_evidence(
            "stage", stage_error, "staged" if staged else "unstaged")
        if failure is not None:
            cleanup_error = None
            # A split stage state means the backend violated its collective contract.  Entering
            # rollback from only the staged ranks would deadlock, so retain evidence and fail.
            if len(set(states)) == 1:
                cleanup = session.rollback if staged else session.abort_prepare
                try:
                    if cleanup() is not None:
                        raise TypeError("scientific writer cleanup must return None")
                except BaseException as error:
                    cleanup_error = error
            else:
                cleanup_error = RuntimeError(
                    "writer stage state differs across ranks; collective cleanup was not entered")
            if cleanup_error is not None:
                _add_exception_note(failure,
                    "async scientific writer cleanup also failed: %s: %s"
                    % (type(cleanup_error).__name__, cleanup_error))
            raise failure

        receipt = None
        published = False
        publish_error = None
        try:
            receipt = session.publish()
            published = True
            if type(receipt) is not OutputPublicationReceipt:
                raise TypeError("async scientific writer publish() must return a real receipt")
            if receipt.selection_identity != frame.request.publication_identity:
                raise ValueError("async writer receipt authenticates another selection")
            if receipt.format != self._format_data["format_name"]:
                raise ValueError(
                    "async writer receipt format differs from its canonical provider")
            expected_parent = target.expanduser().resolve().parent
            if Path(receipt.path).expanduser().resolve().parent != expected_parent:
                raise ValueError(
                    "async writer primary receipt escaped its authenticated target directory")
        except BaseException as error:
            publish_error = error
        failure, _states = self._phase_evidence(
            "publish", publish_error, "published" if published else "unpublished")
        if failure is not None:
            cleanup_error = None
            try:
                if session.rollback() is not None:
                    raise TypeError("scientific writer rollback must return None")
            except BaseException as error:
                cleanup_error = error
            if cleanup_error is not None:
                _add_exception_note(failure,
                    "async scientific writer rollback also failed: %s: %s"
                    % (type(cleanup_error).__name__, cleanup_error))
            raise failure
        if type(receipt) is not OutputPublicationReceipt:
            raise RuntimeError("async writer publication lost its authenticated receipt")
        # Publication is now durable and authenticated.  Writer finalization is release-only: a
        # failure is reported on the real receipt and must never trigger rollback or frame retry.
        finalize_error = None
        try:
            result = session.finalize()
            if result is not None:
                raise TypeError("scientific writer finalize() must return None")
        except BaseException as error:
            finalize_error = "%s: %s" % (type(error).__name__, error)
        return ObserverReceipt(frame.identity, self.authority["provider_id"], {
            "path": Path(receipt.path).resolve().as_posix(),
            "format": receipt.format,
            "output_identity": receipt.output_identity.token,
            "selection_identity": receipt.selection_identity.token,
            "writer_finalize_error": finalize_error,
        })

    def finalize(self) -> None:
        if self._finalized:
            return None
        self._finalized = True
        return None

    def abort(self) -> None:
        self._finalized = True
        return None


def _relative_target(value: Any, *, where: str) -> str:
    result = _text(value, where)
    if Path(result).is_absolute():
        raise ValueError("%s must be a relative output target" % where)
    pieces = result.split("/")
    if any(piece in {"", ".", ".."} for piece in pieces):
        raise ValueError("%s must be a canonical relative output target" % where)
    if PurePosixPath(result).suffix:
        raise ValueError(
            "%s is a logical target and must not contain a file suffix; "
            "the selected provider owns its extension" % where)
    return result


class AsyncScientificOutput(Descriptor):
    """Write exact scientific files on a bounded post-commit worker.

    SERIAL and gathered ROOT writers need no worker MPI.  PER_RANK and COLLECTIVE writers execute
    on one duplicated MPI lane per consumer, isolated from numerical collectives.  The default
    queue is process-lifetime only; a ``DurableJournal`` policy adds the explicit crash-replay
    handoff.
    """

    category = "async_scientific_output"

    def __init__(
        self,
        *,
        format: Any,
        schedule: Any,
        fields: Any,
        levels: Any = None,
        target: Any,
        queue_capacity: Any = 1,
        max_attempts: Any = 1,
        on_failure: Any = None,
        durability: Any = None,
    ) -> None:
        from ._consumer_contracts import ParallelMode
        from ._durable_journal import DurableJournal

        observer = _AsyncScientificWriterObserver(format)
        selected_mode = ParallelMode(observer.consumer_data()["format"]["parallel_mode"])
        if type(schedule) is not Schedule:
            raise TypeError("AsyncScientificOutput.schedule must be an exact pops.time.Schedule")
        field_rows = tuple(fields)
        if not field_rows:
            raise ValueError("AsyncScientificOutput requires at least one field")
        if any(not isinstance(reference, Handle) for reference in field_rows):
            raise TypeError("AsyncScientificOutput fields must contain declaration Handles")
        if any(reference.kind not in _LIVE_FIELD_KINDS for reference in field_rows):
            raise TypeError("AsyncScientificOutput fields accept only state, field, or aux Handles")
        if len(set(field_rows)) != len(field_rows):
            raise ValueError("AsyncScientificOutput fields must be unique")
        selected_levels = AllLevels() if levels is None else levels
        if not isinstance(selected_levels, LevelSelection):
            raise TypeError("AsyncScientificOutput levels must be a typed LevelSelection")
        if isinstance(queue_capacity, bool) or type(queue_capacity) is not int \
                or queue_capacity < 1:
            raise ValueError("AsyncScientificOutput.queue_capacity must be an integer >= 1")
        if isinstance(max_attempts, bool) or type(max_attempts) is not int \
                or max_attempts < 1:
            raise ValueError("AsyncScientificOutput.max_attempts must be an integer >= 1")
        if selected_mode in (ParallelMode.PER_RANK, ParallelMode.COLLECTIVE) \
                and max_attempts != 1:
            raise ValueError(
                "MPI async scientific output requires max_attempts=1; retrying an entered "
                "collective publication is not safe")
        selected_failure = RaiseOnFlush() if on_failure is None else on_failure
        if type(selected_failure) not in _LIVE_FAILURE_POLICIES:
            raise TypeError(
                "AsyncScientificOutput.on_failure must be RaiseOnFlush() or ReportOnly()")
        if durability is not None and type(durability) is not DurableJournal:
            raise TypeError(
                "AsyncScientificOutput.durability must be DurableJournal() or None")
        self.format = format
        self.schedule = schedule
        self.fields = field_rows
        self.levels = selected_levels
        self.target = _relative_target(target, where="AsyncScientificOutput.target")
        self.queue_capacity = queue_capacity
        self.max_attempts = max_attempts
        self.on_failure = selected_failure
        self.durability = durability
        self._operation = _LiveObserverOperation(
            observer,
            selected_mode,
            queue_capacity,
            max_attempts,
            selected_failure,
            durability,
        )

    def declaration_references(self) -> tuple[Handle, ...]:
        return self.fields

    def consumer_authoring(self) -> tuple[Any, ...]:
        from ._consumer_authoring import ConsumerAuthoringNode
        from ._consumer_contracts import ConsumerKind, FailRun

        return (ConsumerAuthoringNode(
            label="async-scientific-output-%s" % self.target.replace("/", "-"),
            kind=ConsumerKind.MONITOR,
            references=self.fields,
            schedule=self.schedule,
            target_uri=self.target,
            output_format=None,
            parallel_mode=self._operation.parallel_mode,
            levels=self.levels,
            operation=self._operation,
            failure_action=FailRun(),
        ),)

    def options(self) -> dict[str, Any]:
        return {
            "format": self._operation.consumer_data()["observer"]["format"],
            "schedule": self.schedule.to_data(),
            "fields": [reference.inspect() for reference in self.fields],
            "levels": self.levels.to_data(),
            "target": self.target,
            "queue_capacity": self.queue_capacity,
            "max_attempts": self.max_attempts,
            "on_failure": self.on_failure.to_data(),
            "durability": (
                None if self.durability is None else self.durability.to_data()),
        }


class LiveVisualization(Descriptor):
    """Stream selected accepted fields to an irreversible post-commit observer.

    Catalyst supports either single-rank delivery or one collective frame per MPI rank on a
    duplicated observer communicator.  ``ROOT`` and ``PER_RANK`` are rejected because Catalyst's
    lifecycle and data handoff are collective.
    """

    category = "monitor"

    def __init__(
        self,
        *,
        observer: Any,
        schedule: Any,
        fields: Any,
        levels: Any = None,
        mode: Any = None,
        queue_capacity: Any = 1,
        max_attempts: Any = 1,
        on_failure: Any = None,
        durability: Any = None,
    ) -> None:
        from ._consumer_contracts import ParallelMode
        from ._durable_journal import DurableJournal

        if not callable(getattr(observer, "consumer_data", None)) \
                or not callable(getattr(observer, "open_session", None)):
            raise TypeError(
                "LiveVisualization observer must implement consumer_data() and open_session()")
        first, second = observer.consumer_data(), observer.consumer_data()
        if type(first) is not dict or type(second) is not dict or first != second:
            raise TypeError("LiveVisualization observer data must be one deterministic dict")
        canonical_bytes(first)
        if type(schedule) is not Schedule:
            raise TypeError("LiveVisualization.schedule must be an exact pops.time.Schedule")
        field_rows = tuple(fields)
        if not field_rows:
            raise ValueError("LiveVisualization requires at least one field")
        if any(not isinstance(reference, Handle) for reference in field_rows):
            raise TypeError("LiveVisualization fields must contain declaration Handles")
        if any(reference.kind not in _LIVE_FIELD_KINDS for reference in field_rows):
            raise TypeError("LiveVisualization fields accept only state, field, or aux Handles")
        if len(set(field_rows)) != len(field_rows):
            raise ValueError("LiveVisualization fields must be unique")
        selected_levels = AllLevels() if levels is None else levels
        if not isinstance(selected_levels, LevelSelection):
            raise TypeError("LiveVisualization levels must be a typed LevelSelection")
        selected_mode = ParallelMode.SERIAL if mode is None else mode
        if type(selected_mode) is not ParallelMode:
            raise TypeError("LiveVisualization.mode must be an exact ParallelMode")
        observer_kind = first.get("observer_kind")
        if selected_mode in (ParallelMode.ROOT, ParallelMode.PER_RANK):
            raise ValueError(
                "LiveVisualization supports only SERIAL or COLLECTIVE mode")
        if selected_mode is ParallelMode.COLLECTIVE and observer_kind != "catalyst":
            raise ValueError(
                "COLLECTIVE LiveVisualization requires the built-in Catalyst observer")
        if isinstance(queue_capacity, bool) or type(queue_capacity) is not int \
                or queue_capacity < 1:
            raise ValueError("LiveVisualization.queue_capacity must be an integer >= 1")
        if isinstance(max_attempts, bool) or type(max_attempts) is not int \
                or max_attempts < 1:
            raise ValueError("LiveVisualization.max_attempts must be an integer >= 1")
        if selected_mode is ParallelMode.COLLECTIVE and max_attempts != 1:
            raise ValueError(
                "MPI Catalyst live visualization requires max_attempts=1; retrying an "
                "entered collective is not safe")
        selected_failure = RaiseOnFlush() if on_failure is None else on_failure
        if type(selected_failure) not in _LIVE_FAILURE_POLICIES:
            raise TypeError(
                "LiveVisualization.on_failure must be RaiseOnFlush() or ReportOnly()")
        if durability is not None and type(durability) is not DurableJournal:
            raise TypeError(
                "LiveVisualization.durability must be DurableJournal() or None")
        operation = _LiveObserverOperation(
            observer, selected_mode, queue_capacity, max_attempts, selected_failure,
            durability)
        operation_data = operation.consumer_data()
        digest = make_identity("live-visualization-declaration", operation_data).hexdigest[:16]
        self.observer = observer
        self.schedule = schedule
        self.fields = field_rows
        self.levels = selected_levels
        self.mode = selected_mode
        self.queue_capacity = queue_capacity
        self.max_attempts = max_attempts
        self.on_failure = selected_failure
        self.durability = durability
        self._operation = operation
        self._target = "live/%s" % digest

    def declaration_references(self) -> tuple[Handle, ...]:
        return self.fields

    def consumer_authoring(self) -> tuple[Any, ...]:
        from ._consumer_authoring import ConsumerAuthoringNode
        from ._consumer_contracts import ConsumerKind, FailRun

        return (ConsumerAuthoringNode(
            label="live-visualization-%s" % self._target.rsplit("/", 1)[-1],
            kind=ConsumerKind.MONITOR,
            references=self.fields,
            schedule=self.schedule,
            target_uri=self._target,
            output_format=None,
            parallel_mode=self.mode,
            levels=self.levels,
            operation=self._operation,
            failure_action=FailRun(),
        ),)

    def options(self) -> dict[str, Any]:
        return {
            "observer": self.observer.consumer_data(),
            "schedule": self.schedule.to_data(),
            "fields": [reference.inspect() for reference in self.fields],
            "levels": self.levels.to_data(),
            "mode": self.mode.value,
            "queue_capacity": self.queue_capacity,
            "max_attempts": self.max_attempts,
            "on_failure": self.on_failure.to_data(),
            "durability": (
                None if self.durability is None else self.durability.to_data()),
        }


__all__ = [
    "AsyncScientificOutput", "Catalyst", "LiveFailurePolicy", "LiveVisualization", "ObserverFrame",
    "ObserverProvider", "ObserverReceipt", "ObserverRun", "ObserverSession", "RaiseOnFlush",
    "ReportOnly", "authenticate_observer_session", "detach_observer_frame",
]
