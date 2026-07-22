"""Public post-commit scientific-output contracts without a native runtime fixture."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from pops._platform_contracts import (
    ExecutionContext,
    ExecutionResource,
    proven_serial_manifest,
)
from pops.identity import Identity, make_identity
from pops.output import (
    AsyncScientificOutput,
    LiveVisualization,
    OutputClock,
    OutputPublicationReceipt,
    ParaView,
    ParallelMode,
    ReportOnly,
    read_paraview_series,
)
from pops.output.observers import ObserverFrame, ObserverRun
from pops.output._writers.common import writer_session_authority
from pops.runtime._observer_runtime import PostCommitObserverQueue
from pops.time import AcceptedStep, Clock, Every, Schedule

from tests.python.unit.output.test_post_commit_observers import _frame


def _serial_context() -> ExecutionContext:
    return ExecutionContext(
        backend=proven_serial_manifest(
            backend="production", target="system", abi="test|c++|c++23", runtime=True),
        communicator=ExecutionResource("communicator", "serial"),
        datatype=ExecutionResource("datatype", "float64"),
        device=ExecutionResource("device", "host"),
    )


def _schedule() -> Schedule:
    clock = Clock("output")
    return Schedule(Every(AcceptedStep(clock), 1))


class _ModeWriterSession:
    def __init__(self, request, target: Path) -> None:
        self.authority = writer_session_authority("mode-writer", request, target)
        self.identity = Identity.from_token(self.authority["session_identity"])
        self._request = request
        self._target = target

    def stage(self):
        return None

    def abort_prepare(self):
        raise AssertionError("successful mode writer must not abort preparation")

    def publish(self):
        self._target.parent.mkdir(parents=True, exist_ok=True)
        self._target.write_bytes(b"published mode fixture\n")
        return OutputPublicationReceipt(
            self._target,
            "mode-writer",
            make_identity("scientific-output", {
                "selection": self._request.publication_identity.token,
                "target": self._target.as_posix(),
            }),
            self._request.publication_identity,
        )

    def rollback(self):
        raise AssertionError("successful mode writer must not roll back")

    def finalize(self):
        return None


class _ModeWriter:
    format = "mode-writer"

    def __init__(self, owner) -> None:
        self._owner = owner

    def preflight(self, _execution_context):
        return {"schema_version": 1, "provider_id": "mode-writer"}

    def prepare_session(self, _snapshot, request, target, *, communicator=None):
        self._owner.communicators.append(communicator)
        return _ModeWriterSession(request, Path(target))


class _ModeFormat:
    __pops_ir_immutable__ = True

    def __init__(self, mode: ParallelMode) -> None:
        self.mode = mode
        self.communicators = []

    def consumer_data(self):
        return {
            "schema_version": 1,
            "provider_id": "pops.test.mode-writer.v1",
            "format_name": "mode-writer",
            "extension": ".mode",
            "parallel_mode": self.mode.value,
        }

    def writer(self):
        return _ModeWriter(self)


@pytest.mark.parametrize(
    ("mode", "uses_lane"),
    [
        (ParallelMode.ROOT, False),
        (ParallelMode.PER_RANK, True),
        (ParallelMode.COLLECTIVE, True),
    ],
)
def test_async_scientific_output_supports_every_distributed_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: ParallelMode,
    uses_lane: bool,
):
    import pops._native_collectives as native_collectives

    frame = _frame(mode=mode)
    format_provider = _ModeFormat(mode)
    descriptor = AsyncScientificOutput(
        format=format_provider,
        schedule=_schedule(),
        fields=(frame.snapshot.fields[0].key.reference,),
        target="distributed-output",
    )
    operation = descriptor.consumer_authoring()[0].operation
    context = _serial_context()
    lane = SimpleNamespace(rank=0, size=2) if uses_lane else None
    monkeypatch.setattr(native_collectives, "rank", lambda communicator: communicator.rank)
    monkeypatch.setattr(native_collectives, "size", lambda communicator: communicator.size)
    monkeypatch.setattr(
        native_collectives,
        "allgather_value",
        lambda communicator, value: tuple(
            dict(value, rank=owner) for owner in range(communicator.size)),
    )
    configuration = {
        "target_uri": "distributed-output",
        "output_root": str(tmp_path),
        "consumer_id": "async-%s" % mode.value,
    }
    if uses_lane:
        configuration["worker_communicator"] = lane
    session = operation.open_runtime_session(configuration, context)

    assert session.authority["threading"] == (
        "dedicated_collective" if uses_lane else "dedicated_serial")
    assert session.authority["worker_mpi"] is uses_lane
    session.initialize(ObserverRun(frame.snapshot.provenance.run_identity))
    receipt = session.execute(frame)
    session.finalize()

    assert receipt.provider_id == "pops.output.async-scientific-writer.v1"
    assert Path(receipt.detail["path"]).read_bytes() == b"published mode fixture\n"
    assert format_provider.communicators == [lane]


def test_collective_async_writer_rejects_rank_divergent_target_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pops._native_collectives as native_collectives

    frame = _frame(mode=ParallelMode.COLLECTIVE)
    descriptor = AsyncScientificOutput(
        format=_ModeFormat(ParallelMode.COLLECTIVE),
        schedule=_schedule(),
        fields=(frame.snapshot.fields[0].key.reference,),
        target="collective-output",
    )
    operation = descriptor.consumer_authoring()[0].operation
    lane = SimpleNamespace(rank=0, size=2)
    monkeypatch.setattr(native_collectives, "rank", lambda communicator: communicator.rank)
    monkeypatch.setattr(native_collectives, "size", lambda communicator: communicator.size)

    def divergent_targets(communicator, value):
        assert communicator is lane
        return (
            dict(value, rank=0),
            dict(value, rank=1, state=value["state"] + "-rank-one"),
        )

    monkeypatch.setattr(native_collectives, "allgather_value", divergent_targets)
    session = operation.open_runtime_session({
        "target_uri": "collective-output",
        "output_root": str(tmp_path),
        "consumer_id": "async-collective-target",
        "worker_communicator": lane,
    }, _serial_context())
    session.initialize(ObserverRun(frame.snapshot.provenance.run_identity))

    with pytest.raises(RuntimeError, match="resolved different target paths"):
        session.execute(frame)


def test_async_paraview_collection_reports_the_real_pvd_primary(tmp_path: Path):
    frame = _frame()
    descriptor = AsyncScientificOutput(
        format=ParaView(ParallelMode.SERIAL, collection=True),
        schedule=_schedule(),
        fields=(frame.snapshot.fields[0].key.reference,),
        target="series",
    )
    operation = descriptor.consumer_authoring()[0].operation
    context = _serial_context()
    operation.preflight(context)
    output_root = tmp_path / "series"
    output_root.mkdir()
    session = operation.open_runtime_session({
        "target_uri": "series",
        "output_root": str(tmp_path),
        "consumer_id": "async-paraview",
    }, context)
    run = ObserverRun(frame.snapshot.provenance.run_identity)

    with PostCommitObserverQueue(
            session, run, consumer_id="async-paraview") as observer_queue:
        observer_queue.submit(frame)
        reports = observer_queue.flush()

    assert reports[0].status == "delivered"
    receipt = reports[0].receipt
    assert receipt is not None
    primary = Path(receipt.detail["path"])
    assert primary.suffix == ".pvd"
    assert primary.is_file()
    assert receipt.detail["writer_finalize_error"] is None
    reopened = read_paraview_series(primary)
    assert reopened.kind == "pvd"
    assert len(reopened.paths) == 1
    assert tuple(output_root.glob("*.vtu"))


def test_async_paraview_logical_target_keeps_every_temporal_leaf(tmp_path: Path):
    first = _frame()
    second = ObserverFrame(
        replace(
            first.snapshot,
            clock=OutputClock.at("macro", 0.5, 5, stage="accepted"),
        ),
        first.request,
    )
    descriptor = AsyncScientificOutput(
        format=ParaView(ParallelMode.SERIAL, collection=True),
        schedule=_schedule(),
        fields=(first.snapshot.fields[0].key.reference,),
        target="nested/result",
    )
    operation = descriptor.consumer_authoring()[0].operation
    context = _serial_context()
    session = operation.open_runtime_session({
        "target_uri": "nested/result",
        "output_root": str(tmp_path),
        "consumer_id": "async-explicit-paraview",
    }, context)
    run = ObserverRun(first.snapshot.provenance.run_identity)

    with PostCommitObserverQueue(
            session, run, consumer_id="async-explicit-paraview") as observer_queue:
        observer_queue.submit(first)
        observer_queue.submit(second)
        reports = observer_queue.flush()

    assert [report.status for report in reports] == ["delivered", "delivered"]
    assert reports[-1].receipt is not None
    latest = read_paraview_series(Path(reports[-1].receipt.detail["path"]))
    assert len(latest.paths) == 2
    assert len(set(latest.paths)) == 2
    assert all(path.is_file() for path in latest.paths)
    assert all(path.parent == tmp_path / "nested/result" for path in latest.paths)


def test_async_scientific_output_target_is_logical_and_format_independent():
    frame = _frame()
    with pytest.raises(ValueError, match="must not contain a file suffix"):
        AsyncScientificOutput(
            format=ParaView(),
            schedule=_schedule(),
            fields=(frame.snapshot.fields[0].key.reference,),
            target="result.vtu",
        )


class _ReleaseFailSession:
    def __init__(self, frame, target: Path, owner) -> None:
        self.authority = writer_session_authority("release-fail", frame.request, target)
        self.identity = Identity.from_token(self.authority["session_identity"])
        self._frame = frame
        self._target = target
        self._owner = owner
        self._staged = False

    def stage(self):
        self._staged = True

    def abort_prepare(self):
        self._owner.abort_calls += 1

    def publish(self):
        self._target.parent.mkdir(parents=True, exist_ok=True)
        self._target.write_text("durable artifact\n")
        return OutputPublicationReceipt(
            self._target,
            "release-fail",
            make_identity("scientific-output", {"frame": self._frame.identity.token}),
            self._frame.request.publication_identity,
        )

    def rollback(self):
        self._owner.rollback_calls += 1
        self._target.unlink(missing_ok=True)

    def finalize(self):
        self._owner.finalize_calls += 1
        raise RuntimeError("release authority unavailable")


class _ReleaseFailWriter:
    format = "release-fail"

    def __init__(self, owner) -> None:
        self._owner = owner

    def preflight(self, _execution_context):
        return {"schema_version": 1, "provider_id": "release-fail", "serial": True}

    def prepare_session(self, snapshot, request, target, *, communicator=None):
        assert communicator is None
        frame = _frame()
        # Preserve the exact request/snapshot supplied by the async worker.
        frame = type(frame)(snapshot, request)
        return _ReleaseFailSession(frame, Path(target), self._owner)


class _ReleaseFailFormat:
    __pops_ir_immutable__ = True

    def __init__(self) -> None:
        self.rollback_calls = 0
        self.abort_calls = 0
        self.finalize_calls = 0

    def consumer_data(self):
        return {
            "schema_version": 1,
            "provider_id": "pops.test.release-fail.v1",
            "format_name": "release-fail",
            "extension": ".artifact",
            "parallel_mode": "serial",
        }

    def writer(self):
        return _ReleaseFailWriter(self)


def test_async_writer_finalize_failure_keeps_durable_receipt_without_rollback(tmp_path: Path):
    frame = _frame()
    format_provider = _ReleaseFailFormat()
    descriptor = AsyncScientificOutput(
        format=format_provider,
        schedule=_schedule(),
        fields=(frame.snapshot.fields[0].key.reference,),
        target="release-output",
        max_attempts=3,
        on_failure=ReportOnly(),
    )
    operation = descriptor.consumer_authoring()[0].operation
    context = _serial_context()
    operation.preflight(context)
    session = operation.open_runtime_session({
        "target_uri": "release-output",
        "output_root": str(tmp_path),
        "consumer_id": "release-failure",
    }, context)
    run = ObserverRun(frame.snapshot.provenance.run_identity)

    observer_queue = PostCommitObserverQueue(
        session, run, consumer_id="release-failure", max_attempts=3)
    observer_queue.submit(frame)
    reports = observer_queue.close()

    assert reports[0].status == "delivered"
    assert reports[0].attempts == 1
    receipt = reports[0].receipt
    assert receipt is not None
    assert "release authority unavailable" in receipt.detail["writer_finalize_error"]
    assert Path(receipt.detail["path"]).is_file()
    assert format_provider.finalize_calls == 1
    assert format_provider.rollback_calls == 0
    assert format_provider.abort_calls == 0


class _WrongAuthorityWriter(_ReleaseFailWriter):
    def prepare_session(self, snapshot, request, target, *, communicator=None):
        assert communicator is None
        frame = type(_frame())(snapshot, request)
        wrong = Path(target).with_name("other.artifact")
        return _ReleaseFailSession(frame, wrong, self._owner)


class _WrongAuthorityFormat(_ReleaseFailFormat):
    def writer(self):
        return _WrongAuthorityWriter(self)


def test_async_writer_refuses_wrong_same_directory_authority_before_io(tmp_path: Path):
    frame = _frame()
    descriptor = AsyncScientificOutput(
        format=_WrongAuthorityFormat(),
        schedule=_schedule(),
        fields=(frame.snapshot.fields[0].key.reference,),
        target="authority-output",
    )
    operation = descriptor.consumer_authoring()[0].operation
    context = _serial_context()
    operation.preflight(context)
    session = operation.open_runtime_session({
        "target_uri": "authority-output",
        "output_root": str(tmp_path),
        "consumer_id": "wrong-authority",
    }, context)
    session.initialize(ObserverRun(frame.snapshot.provenance.run_identity))

    with pytest.raises(ValueError, match="authority differs"):
        session.execute(frame)

    assert tuple(tmp_path.rglob("*.artifact")) == ()


class _ExternalLikeFormat:
    __pops_ir_immutable__ = True

    def consumer_data(self):
        return {
            "schema_version": 1,
            "provider_id": "pops.output.external-writer.v1",
            "format_name": "external-writer",
            "extension": ".native",
            "parallel_mode": "serial",
        }

    def writer(self):
        raise AssertionError("external native writer factory must not be opened")


def test_async_output_refuses_external_writer_before_a_run():
    frame = _frame()
    with pytest.raises(ValueError, match="does not accept ExternalWriter"):
        AsyncScientificOutput(
            format=_ExternalLikeFormat(),
            schedule=_schedule(),
            fields=(frame.snapshot.fields[0].key.reference,),
            target="external",
        )


class _MutableObserver:
    def __init__(self) -> None:
        self.version = 1

    def consumer_data(self):
        return {
            "schema_version": 1,
            "provider_id": "pops.test.mutable-observer.v1",
            "observer_kind": "test",
            "version": self.version,
        }

    def open_session(self, _execution_context):
        raise AssertionError("mutated observer must be refused before session open")


def test_live_operation_refuses_mutated_provider_after_manifest_identity():
    frame = _frame()
    provider = _MutableObserver()
    descriptor = LiveVisualization(
        observer=provider,
        schedule=_schedule(),
        fields=(frame.snapshot.fields[0].key.reference,),
    )
    operation = descriptor.consumer_authoring()[0].operation
    provider.version = 2

    with pytest.raises(RuntimeError, match="changed after its declaration"):
        operation.preflight(_serial_context())
