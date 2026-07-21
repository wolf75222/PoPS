"""Private adapter from accepted consumer effects to exact format writers."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pops._native_collectives import (
    allgather_value,
    rank as native_rank,
    require_world,
    size as native_size,
)
from pops.identity import Identity, canonical_bytes, make_identity
from pops.output._consumer_contracts import ParallelMode
from pops.output.data import OutputRequest, OutputSnapshot
from pops.output.provider import consumer_format_data
from pops.output._writers.common import (
    authenticate_series_catalog,
    authenticate_series_publication,
    authenticate_writer_session,
)

from ._consumer import (
    AcceptedSideEffect,
    ConsumerPublisher,
    PreparedPublication,
    PublicationReceipt,
)


def _resolved_series_catalog(
    provider: Any,
    format_data: Mapping[str, Any],
) -> tuple[Any, dict[str, Any] | None]:
    options = format_data.get("options", {})
    if not isinstance(options, Mapping):
        raise TypeError("scientific output format options must be a mapping")
    series_enabled = options.get("series", False)
    if type(series_enabled) is not bool:
        raise TypeError("scientific output format series option must be an exact bool")
    factory = getattr(provider, "series_catalog", None)
    if factory is not None and not callable(factory):
        raise TypeError("scientific output format series_catalog must be callable")
    catalog = None if factory is None else factory()
    if series_enabled:
        if catalog is None:
            raise TypeError("series-enabled output format provides no catalogue capability")
        return catalog, authenticate_series_catalog(catalog, format_data)
    if catalog is not None:
        raise ValueError("series-disabled output format returned a catalogue capability")
    return None, None


def preflight_consumer_publication(graph: Any, execution_context: Any) -> None:
    """Validate installed scientific-output capabilities before native engine creation."""
    from pops.output._consumer_contracts import ConsumerGraph, ConsumerKind
    from pops._platform_contracts import ExecutionContext

    if type(graph) is not ConsumerGraph:
        raise TypeError("output preflight requires the exact resolved ConsumerGraph")
    if type(execution_context) is not ExecutionContext:
        raise TypeError("output preflight requires an exact ExecutionContext")
    communicator = execution_context.communicator
    serial = communicator.identity == "serial"
    handle = communicator.handle
    if serial and handle is not None:
        raise ValueError("serial output preflight rejects a hidden communicator handle")
    native = None if serial else require_world(handle)

    rank, size = 0, 1
    capability_rows: list[dict[str, Any]] = []
    local_error = None
    graph_token = None
    try:
        graph_token = graph.identity.token
        if not serial:
            rank, size = native_rank(native), native_size(native)
        for manifest in graph.nodes:
            if manifest.kind is not ConsumerKind.SCIENTIFIC_OUTPUT:
                continue
            mode = manifest.parallel_mode
            if mode is ParallelMode.SERIAL and not serial:
                raise ValueError(
                    "SERIAL scientific output requires a serial ExecutionContext")
            if mode is not ParallelMode.SERIAL and serial:
                raise ValueError(
                    "%s scientific output requires a distributed ExecutionContext" % mode.name)
            format_data = manifest.output_format_data
            if format_data is None:
                raise ValueError("scientific output has no resolved format authority")
            if format_data["parallel_mode"] != mode.value:
                raise ValueError("scientific output format mode differs from its manifest")
            capability: dict[str, Any] = {
                "consumer": manifest.qualified_id,
                "mode": mode.value,
                "format": dict(format_data),
            }
            writer_factory = getattr(manifest.output_format, "writer", None)
            if not callable(writer_factory):
                raise TypeError("scientific output format has no structural writer() protocol")
            writer = writer_factory()
            writer_preflight = getattr(writer, "preflight", None)
            if not callable(writer_preflight):
                raise TypeError(
                    "scientific output writer has no preflight(execution_context) protocol")
            writer_capability = writer_preflight(execution_context)
            if type(writer_capability) is not dict:
                raise TypeError("scientific output writer preflight must return an exact dict")
            canonical_bytes(writer_capability)
            capability["writer"] = writer_capability
            _catalog, catalog_authority = _resolved_series_catalog(
                manifest.output_format, format_data)
            capability["series_catalog"] = catalog_authority
            capability_rows.append(capability)
    except BaseException as exc:
        local_error = "%s: %s" % (type(exc).__name__, exc)

    authority = {
        "rank": rank,
        "size": size,
        "graph": graph_token,
        "capabilities": capability_rows,
        "error": local_error,
    }
    if serial:
        if local_error is not None:
            raise RuntimeError("scientific-output preflight failed: " + local_error)
        return

    rows = allgather_value(native, authority)
    if len(rows) != size or any(
        not isinstance(row, Mapping)
        or set(row) != {"rank", "size", "graph", "capabilities", "error"}
        for row in rows
    ):
        raise RuntimeError("scientific-output preflight returned malformed rank authority")
    failures = [
        "rank %d: %s" % (owner, row["error"])
        for owner, row in enumerate(rows) if row["error"] is not None
    ]
    if failures:
        raise RuntimeError(
            "scientific-output preflight failed across ranks: " + "; ".join(failures))
    if any(row["rank"] != owner or row["size"] != size
           for owner, row in enumerate(rows)):
        raise RuntimeError("scientific-output preflight rank topology is not canonical")
    canonical = {key: rows[0][key] for key in ("size", "graph", "capabilities")}
    if any(
        any(row[key] != canonical[key] for key in canonical)
        for row in rows[1:]
    ):
        raise RuntimeError(
            "scientific-output graph/capability authority differs across ranks")


def _request_family(request: OutputRequest) -> dict[str, Any]:
    data = request.to_data()
    data.pop("rank")
    return data


def _snapshot_family(preparation: OutputPreparation) -> dict[str, Any]:
    data = preparation.snapshot.to_data(preparation.request)
    data["selection"] = _request_family(preparation.request)
    data["fields"] = [dict(row, pieces=[]) for row in data["fields"]]
    return data


def _per_rank_target_family(target: Any, rank: int) -> str:
    path = Path(target).expanduser().resolve()
    explicit = ".rank%06d" % rank
    deterministic = "__r%06d__" % rank
    matches = path.name.count(explicit) + path.name.count(deterministic)
    if matches != 1:
        raise ValueError(
            "PER_RANK output target must carry exactly one canonical rank qualifier")
    name = path.name.replace(explicit, ".rank{rank}").replace(
        deterministic, "__r{rank}__")
    return str(path.with_name(name))


def _authenticate_preparation(
    effect: AcceptedSideEffect,
    preparation: OutputPreparation,
    format_data: Mapping[str, Any],
) -> None:
    """Prove the cross-rank request/format/target authority before a writer starts."""
    communicator = preparation.communicator
    if communicator is None:
        return
    envelope: dict[str, Any]
    try:
        target = str(Path(preparation.target).expanduser().resolve())
        family = (
            _per_rank_target_family(preparation.target, preparation.request.rank)
            if preparation.request.parallel_mode is ParallelMode.PER_RANK else target
        )
        envelope = {
            "error": None,
            "rank": preparation.request.rank,
            "size": preparation.request.size,
            "mode": preparation.request.parallel_mode.value,
            "effect": effect.identity.token,
            "payload": effect.payload.identity.token,
            "format": dict(format_data),
            "request_family": _request_family(preparation.request),
            "snapshot_family": _snapshot_family(preparation),
            "target": target,
            "target_family": family,
        }
    except BaseException as exc:
        envelope = {
            "error": "%s: %s" % (type(exc).__name__, exc),
            "rank": preparation.request.rank,
            "size": preparation.request.size,
            "mode": preparation.request.parallel_mode.value,
            "effect": None,
            "payload": None,
            "format": None,
            "request_family": None,
            "snapshot_family": None,
            "target": None,
            "target_family": None,
        }
    rows = allgather_value(communicator, envelope)
    required = {
        "error", "rank", "size", "mode", "effect", "payload", "format",
        "request_family", "snapshot_family", "target", "target_family",
    }
    if len(rows) != preparation.request.size or any(
        not isinstance(row, Mapping) or set(row) != required
        for row in rows
    ):
        raise RuntimeError("output preflight returned an incomplete rank authority")
    failures = [
        "rank %d: %s" % (rank, row["error"])
        for rank, row in enumerate(rows) if row["error"] is not None
    ]
    if failures:
        raise RuntimeError("output preparation preflight failed: " + "; ".join(failures))
    if any(row["rank"] != rank for rank, row in enumerate(rows)):
        raise RuntimeError("output preflight rank order differs from communicator order")
    authority = rows[0]
    shared_keys = (
        "size", "mode", "effect", "payload", "format", "request_family",
        "snapshot_family", "target_family",
    )
    mismatches = [
        rank for rank, row in enumerate(rows)
        if any(row[key] != authority[key] for key in shared_keys)
    ]
    if mismatches:
        raise RuntimeError(
            "output request/format/target authority differs across ranks: "
            + ", ".join(map(str, mismatches)))
    mode = preparation.request.parallel_mode
    targets = tuple(row["target"] for row in rows)
    if mode in (ParallelMode.ROOT, ParallelMode.COLLECTIVE):
        if any(target != targets[0] for target in targets[1:]):
            raise RuntimeError("shared output mode resolved different targets across ranks")
    elif mode is ParallelMode.PER_RANK and len(set(targets)) != len(targets):
        raise RuntimeError("PER_RANK output did not resolve one distinct target per rank")

@dataclass(frozen=True, slots=True)
class OutputPreparation:
    """Exact writer input resolved from one already accepted side effect."""

    format: Any
    snapshot: OutputSnapshot
    request: OutputRequest
    target: Any
    communicator: Any = None

    def __post_init__(self) -> None:
        format_data = consumer_format_data(
            self.format, where="OutputPreparation.format")
        if type(self.snapshot) is not OutputSnapshot or type(self.request) is not OutputRequest:
            raise TypeError("output preparation requires exact snapshot/request values")
        if self.request.parallel_mode.value != format_data["parallel_mode"]:
            raise ValueError("resolved output request parallel mode differs from its format")
        if self.request.parallel_mode is ParallelMode.SERIAL:
            if self.communicator is not None:
                raise ValueError("SERIAL output cannot carry a communicator")
        else:
            native = require_world(self.communicator)
            if (native_rank(native), native_size(native)) != (
                    self.request.rank, self.request.size):
                raise ValueError("OutputRequest rank/size differs from its communicator")


class PreparedConsumerOutput(PreparedPublication):
    """Bind all-rank structural sessions to one distributed publication receipt."""

    __slots__ = (
        "_effect", "_session", "_publisher_id", "_request", "_communicator",
        "_target_path", "_format_name", "_series_publication", "_series_authority",
        "_series_finalized", "_writer_finalized", "_artifact_authority",
    )

    def __init__(self, effect: AcceptedSideEffect, session: Any,
                 publisher_id: str, request: OutputRequest, communicator: Any,
                 *, target: Any, format_name: str,
                 series_publication: Any = None,
                 series_publication_authority: dict[str, Any] | None = None) -> None:
        self._effect = effect
        self._session = session
        self._publisher_id = publisher_id
        self._request = request
        self._communicator = communicator
        self._target_path = Path(target).expanduser().resolve()
        self._format_name = format_name
        if series_publication is None and series_publication_authority is not None:
            raise ValueError("series authority requires a prepared publication")
        self._series_publication = series_publication
        self._series_authority = (
            None if series_publication_authority is None
            else dict(series_publication_authority))
        self._series_finalized = series_publication is None
        self._writer_finalized = False
        self._artifact_authority = None
        authenticate_writer_session(session)

    @property
    def effect_identity(self):
        return self._effect.identity

    @property
    def payload_identity(self):
        return self._effect.payload.identity

    @property
    def temporary(self):
        return getattr(self._session, "temporary", None)

    @property
    def target(self):
        return self._target_path

    @property
    def recoveries(self) -> tuple[Any, ...]:
        recoveries = getattr(self._session, "recoveries", ()) \
            if self._session is not None else ()
        if type(recoveries) is not tuple:
            raise TypeError("writer session recoveries must be a tuple")
        return recoveries

    def _allgather(self, envelope: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        if self._communicator is None:
            return (envelope,)
        rows = allgather_value(self._communicator, envelope)
        if len(rows) != self._request.size:
            raise RuntimeError("output communicator returned an incomplete rank envelope")
        return rows

    @staticmethod
    def _receipt_data(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        output_identity = getattr(value, "output_identity", None)
        selection_identity = getattr(value, "selection_identity", None)
        path = getattr(value, "path", None)
        format_name = getattr(value, "format", None)
        file_evidence = getattr(value, "file_evidence", None)
        if type(output_identity) is not Identity or type(selection_identity) is not Identity:
            raise TypeError("output receipt must carry exact output/selection identities")
        if path is None:
            raise TypeError("output receipt path must be filesystem-bounded")
        try:
            resolved_path = Path(path).expanduser().resolve().as_posix()
        except TypeError as exc:
            raise TypeError("output receipt path must be filesystem-bounded") from exc
        if not isinstance(format_name, str) or not format_name \
                or format_name.strip() != format_name:
            raise TypeError("output receipt format must be canonical text")
        if file_evidence is not None and (
            not isinstance(file_evidence, (list, tuple))
            or len(file_evidence) != 5
            or any(isinstance(item, bool) or type(item) is not int or item < 0
                   for item in file_evidence)
        ):
            raise TypeError(
                "output receipt file_evidence must be exact regular-file evidence or None")
        return {
            "output_identity": output_identity.token,
            "selection_identity": selection_identity.token,
            "path": resolved_path,
            "format": format_name,
            "file_evidence": (
                None if file_evidence is None else list(file_evidence)),
        }

    @staticmethod
    def _failure(operation: str, rows: tuple[dict[str, Any], ...]) -> None:
        failures = [
            "rank %d: %s" % (rank, row["error"])
            for rank, row in enumerate(rows) if row["error"] is not None
        ]
        if failures:
            raise RuntimeError(
                "%s output %s failed: %s"
                % (rows[0]["mode"].upper(), operation, "; ".join(failures)))

    def _operate(self, operation: str) -> tuple[dict[str, Any], ...]:
        result = error = None
        try:
            value = getattr(self._session, operation)()
            if operation == "publish":
                result = self._receipt_data(value)
            elif value is not None:
                raise TypeError("writer session %s() must return None" % operation)
        except BaseException as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
        rows = self._allgather({
            "mode": self._request.parallel_mode.value,
            "rank": self._request.rank,
            "result": result,
            "error": error,
        })
        self._failure(operation, rows)
        return rows

    def publish(self) -> PublicationReceipt:
        rows = self._operate("publish")
        mode = self._request.parallel_mode
        signatures = []
        for rank, row in enumerate(rows):
            local = row["result"]
            if local is None:
                signatures.append(None)
                continue
            if not isinstance(local, Mapping) or set(local) != {
                "output_identity", "selection_identity", "path", "format", "file_evidence",
            }:
                raise TypeError("rank %d output receipt has an invalid schema" % rank)
            output_identity = Identity.from_token(local["output_identity"])
            selection_identity = Identity.from_token(local["selection_identity"])
            if selection_identity != self._request.publication_identity:
                raise ValueError("output receipt selection identity differs from its request")
            try:
                path = Path(local["path"]).expanduser().resolve()
            except TypeError as exc:
                raise TypeError("output receipt path must be filesystem-bounded") from exc
            format_name = local["format"]
            if format_name != self._format_name:
                raise ValueError("output receipt format differs from its canonical provider")
            evidence_data = local["file_evidence"]
            if self._series_publication is not None and evidence_data is None:
                raise TypeError(
                    "series-enabled output receipt must carry exact regular-file evidence")
            file_evidence = (
                None if evidence_data is None else tuple(evidence_data))
            signatures.append((
                output_identity.token,
                selection_identity.token,
                path.as_posix(),
                format_name,
                file_evidence,
            ))
        if mode is ParallelMode.COLLECTIVE:
            if not signatures or any(value != signatures[0] for value in signatures):
                raise RuntimeError(
                    "COLLECTIVE output ranks returned different shared artifact receipts")
            rows = (dict(rows[0], rank=0),)
        local_artifacts = []
        for row in rows:
            local = row["result"]
            if local is None:
                continue
            path = Path(local["path"]).expanduser().resolve()
            output_identity = Identity.from_token(local["output_identity"])
            selection_identity = Identity.from_token(local["selection_identity"])
            artifact = make_identity("scientific-output-rank-artifact", {
                "parallel_mode": self._request.parallel_mode.value,
                "rank": row["rank"],
                "output_identity": output_identity.to_data(),
                "selection_identity": selection_identity.to_data(),
                "target": path.as_posix(),
                "format": local["format"],
                "file_evidence": local["file_evidence"],
            })
            local_artifacts.append((int(row["rank"]), artifact.token))
        if mode is ParallelMode.PER_RANK:
            expected = tuple(range(self._request.size))
            if tuple(rank for rank, _ in local_artifacts) != expected:
                raise RuntimeError("PER_RANK output did not publish one artifact per rank")
            aggregate = make_identity("scientific-output-artifact-set", {
                "parallel_mode": mode.value,
                "rank_artifacts": [
                    {"rank": rank, "artifact_id": artifact}
                    for rank, artifact in local_artifacts
                ],
            }).token
        else:
            if len(local_artifacts) != 1 or local_artifacts[0][0] != 0:
                raise RuntimeError(
                    "%s output did not produce exactly one shared rank-0 artifact" % mode.name)
            aggregate = local_artifacts[0][1]
        if self._series_publication is not None:
            root_artifact = rows[0]["result"]
            if not isinstance(root_artifact, Mapping):
                raise RuntimeError("series output has no shared rank-zero artifact authority")
            self._artifact_authority = dict(root_artifact)
        return PublicationReceipt(
            self.effect_identity,
            self.payload_identity,
            self._publisher_id,
            aggregate,
            mode,
            tuple(local_artifacts),
        )

    def discard(self) -> None:
        self._operate("abort_prepare")

    def rollback(self) -> None:
        self._operate("rollback")

    def finalize(self) -> None:
        failures = []
        if not self._series_finalized:
            result = error = None
            if self._request.rank == 0:
                try:
                    series_publication = self._series_publication
                    if series_publication is None:
                        raise RuntimeError(
                            "scientific output series has no prepared publication")
                    if self._series_authority is not None \
                            and series_publication.authority != self._series_authority:
                        raise RuntimeError(
                            "scientific output series publication authority changed")
                    if self._artifact_authority is None:
                        raise RuntimeError(
                            "scientific output series has no published artifact authority")
                    result = series_publication.publish(self._artifact_authority)
                    if result is not None:
                        raise TypeError(
                            "scientific output series publication publish() must return None")
                except BaseException as exc:
                    error = "%s: %s" % (type(exc).__name__, exc)
            rows = self._allgather({
                "mode": self._request.parallel_mode.value,
                "rank": self._request.rank,
                "result": result,
                "error": error,
            })
            try:
                self._failure("series finalization", rows)
            except BaseException as exc:
                failures.append(exc)
            else:
                self._series_finalized = True
                self._series_publication = None
                self._series_authority = None
                self._artifact_authority = None
        if not self._writer_finalized:
            try:
                self._operate("finalize")
            except BaseException as exc:
                failures.append(exc)
            else:
                self._writer_finalized = True
                self._session = None
        if failures:
            primary = failures[0]
            for secondary in failures[1:]:
                add_note = getattr(primary, "add_note", None)
                if callable(add_note):
                    add_note("additional output finalization failure: %s" % secondary)
            raise primary


class ConsumerOutputPublisher(ConsumerPublisher):
    """Dispatch only accepted scientific-output effects to their exact writer."""

    __slots__ = ("_resolve", "_retain_recoveries", "publisher_id")

    def __init__(self, resolve: Callable[[AcceptedSideEffect], OutputPreparation], *,
                 retain_recoveries: Callable[[tuple[Any, ...]], tuple[str, ...]] | None = None,
                 publisher_id: str = "pops.exact-output.v1") -> None:
        if not callable(resolve):
            raise TypeError("ConsumerOutputPublisher resolver must be callable")
        if retain_recoveries is not None and not callable(retain_recoveries):
            raise TypeError("ConsumerOutputPublisher recovery retainer must be callable")
        if not isinstance(publisher_id, str) or not publisher_id or publisher_id.strip() != publisher_id:
            raise TypeError("ConsumerOutputPublisher publisher_id must be canonical text")
        self._resolve = resolve
        self._retain_recoveries = retain_recoveries
        self.publisher_id = publisher_id

    def _register_stage_recoveries(self, session: Any) -> tuple[str, ...]:
        """Transfer abort-created authorities into the bound RuntimeInstance immediately."""
        try:
            recoveries = getattr(session, "recoveries", ())
            if type(recoveries) is not tuple:
                raise TypeError("writer session recoveries must be a tuple")
        except BaseException as error:
            return ("stage recovery inspection failed: %s: %s" % (
                type(error).__name__, error),)
        if not recoveries:
            return ()
        if self._retain_recoveries is None:
            return ("stage recovery registry is unavailable",)
        try:
            result = self._retain_recoveries(recoveries)
            if type(result) is not tuple or any(
                    not isinstance(item, str) or not item for item in result):
                raise TypeError("stage recovery registry must return tuple[str, ...]")
            return result
        except BaseException as error:
            return ("stage recovery registration failed: %s: %s" % (
                type(error).__name__, error),)

    def prepare(self, effect: AcceptedSideEffect) -> PreparedConsumerOutput:
        if type(effect) is not AcceptedSideEffect:
            raise TypeError("ConsumerOutputPublisher requires an exact AcceptedSideEffect")
        preparation = self._resolve(effect)
        if type(preparation) is not OutputPreparation:
            raise TypeError("output effect resolver must return an exact OutputPreparation")
        if preparation.request.consumer_id != effect.consumer_id:
            raise ValueError("output request consumer identity differs from its accepted effect")
        target_format = effect.target.output_format
        if not isinstance(target_format, Mapping):
            raise TypeError("accepted output target must carry a resolved format mapping")
        if consumer_format_data(
                preparation.format, where="resolved output format") != dict(target_format):
            raise ValueError("resolved output format differs from its accepted target")
        mode = effect.target.parallel_mode
        if preparation.request.parallel_mode is not mode:
            raise ValueError("resolved output parallel mode differs from its accepted target")
        writer = None
        series_catalog = None
        catalog_authority = None
        series_publication = None
        series_publication_authority = None
        writer_error = None
        try:
            writer = preparation.format.writer()
            if not callable(getattr(writer, "preflight", None)) \
                    or not callable(getattr(writer, "prepare_session", None)):
                raise TypeError(
                    "scientific output writer must implement preflight() and prepare_session()")
            series_catalog, catalog_authority = _resolved_series_catalog(
                preparation.format, dict(target_format))
            if series_catalog is not None:
                if catalog_authority is None:
                    raise RuntimeError(
                        "scientific output series catalogue has no authenticated authority")
                series_publication = series_catalog.prepare(
                    preparation.target,
                    preparation.snapshot,
                    preparation.request,
                )
                series_publication_authority = authenticate_series_publication(
                    series_publication,
                    catalog_authority,
                    target=preparation.target,
                    snapshot=preparation.snapshot,
                    request=preparation.request,
                )
        except BaseException as exc:
            writer_error = "%s: %s" % (type(exc).__name__, exc)
        if preparation.communicator is not None:
            writer_rows = allgather_value(preparation.communicator, {
                "rank": preparation.request.rank,
                "error": writer_error,
                "catalog": catalog_authority,
                "series_publication": series_publication_authority,
            })
            if len(writer_rows) != preparation.request.size or any(
                not isinstance(row, Mapping)
                or set(row) != {
                    "rank", "error", "catalog", "series_publication",
                }
                or row["rank"] != rank
                for rank, row in enumerate(writer_rows)
            ):
                raise RuntimeError("output writer factory returned malformed rank authority")
            writer_failures = [
                "rank %d: %s" % (rank, row["error"])
                for rank, row in enumerate(writer_rows) if row["error"] is not None
            ]
            if writer_failures:
                raise RuntimeError(
                    "output writer factory failed across ranks: " + "; ".join(writer_failures))
            if any(row["catalog"] != writer_rows[0]["catalog"] for row in writer_rows[1:]):
                raise RuntimeError("output series catalogue authority differs across ranks")
            if any(
                row["series_publication"]
                != writer_rows[0]["series_publication"]
                for row in writer_rows[1:]
            ):
                raise RuntimeError(
                    "output series publication authority differs across ranks")
        elif writer_error is not None:
            raise RuntimeError("SERIAL output writer factory failed: " + writer_error)
        if writer is None:
            raise RuntimeError("output writer factory returned no writer")
        _authenticate_preparation(effect, preparation, dict(target_format))

        session = None
        authority = None
        error = None
        try:
            session = writer.prepare_session(
                preparation.snapshot,
                preparation.request,
                preparation.target,
                communicator=preparation.communicator,
            )
            authority = authenticate_writer_session(session)
            expected_target = Path(preparation.target).expanduser().resolve().as_posix()
            expected = {
                "format": target_format["format_name"],
                "parallel_mode": mode.value,
                "rank": preparation.request.rank,
                "size": preparation.request.size,
                "target": expected_target,
                "selection_identity": preparation.request.publication_identity.token,
            }
            if any(authority[key] != value for key, value in expected.items()):
                raise ValueError(
                    "writer session authority differs from its resolved request/target")
        except BaseException as exc:
            error = "%s: %s" % (type(exc).__name__, exc)

        if preparation.communicator is not None:
            rows = allgather_value(preparation.communicator, {
                "rank": preparation.request.rank,
                "error": error,
                "authority": authority,
            })
            if len(rows) != preparation.request.size or any(
                not isinstance(row, Mapping)
                or set(row) != {"rank", "error", "authority"}
                or row["rank"] != rank
                for rank, row in enumerate(rows)
            ):
                raise RuntimeError(
                    "%s output session returned malformed rank authority" % mode.name)
            failures = [
                "rank %d: %s" % (rank, row["error"])
                for rank, row in enumerate(rows) if row["error"] is not None
            ]
            if failures:
                raise RuntimeError(
                    "%s output session preparation failed: %s"
                    % (mode.name, "; ".join(failures)))
            session_authorities = tuple(row["authority"] for row in rows)
            shared_keys = ("schema_version", "format", "parallel_mode", "size")
            if mode is not ParallelMode.PER_RANK:
                shared_keys += ("selection_identity",)
            if any(
                any(row[key] != session_authorities[0][key] for key in shared_keys)
                for row in session_authorities[1:]
            ):
                raise RuntimeError("output writer session authority differs across ranks")
            session_targets = tuple(row["target"] for row in session_authorities)
            if mode is ParallelMode.PER_RANK:
                if len(set(session_targets)) != preparation.request.size:
                    raise RuntimeError(
                        "PER_RANK writer sessions do not own distinct local targets")
            elif any(target != session_targets[0] for target in session_targets[1:]):
                raise RuntimeError("shared writer sessions do not own one target")
        elif error is not None:
            raise RuntimeError("SERIAL output session preparation failed: %s" % error)
        if session is None:
            raise RuntimeError("output writer session preparation returned no session")

        stage_error = None
        try:
            result = session.stage()
            if result is not None:
                raise TypeError("writer session stage() must return None")
        except BaseException as exc:
            stage_error = "%s: %s" % (type(exc).__name__, exc)
        if preparation.communicator is not None:
            stage_rows = allgather_value(preparation.communicator, {
                "rank": preparation.request.rank,
                "error": stage_error,
            })
            if len(stage_rows) != preparation.request.size or any(
                not isinstance(row, Mapping)
                or set(row) != {"rank", "error"}
                or row["rank"] != rank
                for rank, row in enumerate(stage_rows)
            ):
                raise RuntimeError("output writer stage returned malformed rank status")
            stage_failures = [
                "rank %d: %s" % (rank, row["error"])
                for rank, row in enumerate(stage_rows) if row["error"] is not None
            ]
        else:
            stage_failures = ([] if stage_error is None else ["rank 0: " + stage_error])
        if stage_failures:
            cleanup_error = None
            try:
                session.abort_prepare()
            except BaseException as exc:
                cleanup_error = "%s: %s" % (type(exc).__name__, exc)
            recovery_failures = self._register_stage_recoveries(session)
            if recovery_failures:
                recovery_error = "recovery transfer: " + "; ".join(recovery_failures)
                cleanup_error = (
                    recovery_error if cleanup_error is None
                    else cleanup_error + "; " + recovery_error
                )
            cleanup_failures = []
            if preparation.communicator is not None:
                cleanup_rows = allgather_value(preparation.communicator, cleanup_error)
                cleanup_failures = [
                    "rank %d: %s" % (rank, failure)
                    for rank, failure in enumerate(cleanup_rows) if failure is not None
                ]
            elif cleanup_error is not None:
                cleanup_failures = ["rank 0: " + cleanup_error]
            detail = "; ".join(stage_failures)
            if cleanup_failures:
                detail += "; cleanup failed: " + "; ".join(cleanup_failures)
            raise RuntimeError("%s output staging failed: %s" % (mode.name, detail))
        return PreparedConsumerOutput(
            effect, session, self.publisher_id,
            preparation.request, preparation.communicator,
            target=preparation.target,
            format_name=target_format["format_name"],
            series_publication=series_publication,
            series_publication_authority=series_publication_authority,
        )


__all__ = [
    "ConsumerOutputPublisher", "OutputPreparation", "PreparedConsumerOutput",
    "preflight_consumer_publication",
]
