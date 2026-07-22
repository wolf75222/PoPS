"""Public durability authoring and the narrow post-commit replay seam."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pops.identity import canonical_bytes, make_identity
from pops.output import (
    AsyncScientificOutput,
    DurableJournal,
    LiveVisualization,
    NPZ,
    ParallelMode,
)
from pops.output._durable_journal import DurableJournal as InternalDurableJournal
from pops.output.observers import ObserverFrame, ObserverReceipt, ObserverRun
from pops.runtime._observer_runtime import (
    PostCommitObserverQueue,
    _detach_owned_observer_frame,
)
from pops.runtime._runtime_consumers import (
    RuntimeConsumerPublisher,
    _PreparedLiveVisualization,
)
from pops.time import AcceptedStep, Clock, Every, Schedule
from tests.python.unit.output.test_post_commit_observers import _frame


def _schedule() -> Schedule:
    return Schedule(Every(AcceptedStep(Clock("durable-observer")), 1))


class _ObserverProvider:
    __pops_ir_immutable__ = True

    def consumer_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider_id": "pops.test.durable-observer.v1",
            "observer_kind": "test",
        }

    def open_session(self, _execution_context: Any) -> _AckSession:
        return _AckSession()


class _AckSession:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.frames: list[ObserverFrame] = []
        self.initialized = False
        self.finalized = False
        self.aborted = False
        self.run: ObserverRun | None = None

    @property
    def authority(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider_id": "pops.test.durable-ack.v1",
            "delivery": "post_commit",
            "threading": "dedicated_serial",
            "worker_mpi": False,
        }

    def initialize(self, run: ObserverRun) -> None:
        assert run.run_identity.domain == "run"
        self.run = run
        self.initialized = True

    def execute(self, frame: ObserverFrame) -> ObserverReceipt:
        self.frames.append(frame)
        if self.fail:
            raise RuntimeError("delivery interrupted before acknowledgement")
        return ObserverReceipt(
            frame.identity,
            self.authority["provider_id"],
            {"archive_acknowledged": True},
        )

    def finalize(self) -> None:
        self.finalized = True

    def abort(self) -> None:
        self.aborted = True


def _authoring_identity(descriptor: Any):
    node = descriptor.consumer_authoring()[0]
    first = node.operation.consumer_data()
    second = node.operation.consumer_data()
    assert first == second
    canonical_bytes(first)
    data = node.canonical_data(lambda reference: reference)
    return first, make_identity("test-consumer-authoring", data)


def test_durable_journal_is_public_and_canonical_in_both_observer_descriptors(
    tmp_path: Path,
) -> None:
    assert DurableJournal is InternalDurableJournal
    frame = _frame()
    field = frame.snapshot.fields[0].key.reference
    journal = DurableJournal(tmp_path / "journal", sync="none", recover="manual")
    expected = {
        "schema_version": 1,
        "kind": "durable_observer_journal",
        "root": journal.root.as_posix(),
        "sync": "none",
        "recover": "manual",
            "delivery": "at_least_once_after_handoff",
    }
    assert journal.to_data() == expected
    canonical_bytes(expected)

    async_output = AsyncScientificOutput(
        format=NPZ(ParallelMode.SERIAL),
        schedule=_schedule(),
        fields=(field,),
        target="durable/npz",
        durability=journal,
    )
    live_output = LiveVisualization(
        observer=_ObserverProvider(),
        schedule=_schedule(),
        fields=(field,),
        mode=ParallelMode.SERIAL,
        durability=journal,
    )

    async_data, async_identity = _authoring_identity(async_output)
    live_data, live_identity = _authoring_identity(live_output)

    assert async_data["durability"] == expected
    assert live_data["durability"] == expected
    assert async_output.options()["durability"] == expected
    assert live_output.options()["durability"] == expected

    async_clone_data, async_clone_identity = _authoring_identity(
        AsyncScientificOutput(
            format=NPZ(ParallelMode.SERIAL),
            schedule=_schedule(),
            fields=(field,),
            target="durable/npz",
            durability=journal,
        )
    )
    live_clone_data, live_clone_identity = _authoring_identity(
        LiveVisualization(
            observer=_ObserverProvider(),
            schedule=_schedule(),
            fields=(field,),
            mode=ParallelMode.SERIAL,
            durability=journal,
        )
    )
    assert async_clone_data == async_data
    assert live_clone_data == live_data
    assert async_clone_identity == async_identity
    assert live_clone_identity == live_identity

    _, volatile_async_identity = _authoring_identity(
        AsyncScientificOutput(
            format=NPZ(ParallelMode.SERIAL),
            schedule=_schedule(),
            fields=(field,),
            target="durable/npz",
        )
    )
    _, volatile_live_identity = _authoring_identity(
        LiveVisualization(
            observer=_ObserverProvider(),
            schedule=_schedule(),
            fields=(field,),
            mode=ParallelMode.SERIAL,
        )
    )
    assert volatile_async_identity != async_identity
    assert volatile_live_identity != live_identity


def test_queue_acknowledges_committed_archive_and_replays_failed_delivery(
    tmp_path: Path,
) -> None:
    root = tmp_path / "journal"
    journal = DurableJournal(root, sync="none", recover="manual")
    direct_frame = _frame(field_name="direct_temperature")
    direct = journal.commit(journal.prepare(direct_frame))
    direct_session = _AckSession()

    with PostCommitObserverQueue(
        direct_session,
        ObserverRun(direct_frame.snapshot.provenance.run_identity),
        consumer_id="durable-direct",
    ) as queue:
        queue.submit(direct_frame, journal=journal, journal_record=direct)
        direct_reports = queue.flush()

    assert direct_session.initialized and direct_session.finalized
    assert [report.status for report in direct_reports] == ["delivered"]
    assert journal.list_pending() == ()
    assert [record.state for record in journal.list_committed()] == ["delivered"]

    replay_frame = _frame(field_name="replayed_temperature")
    pending = journal.commit(journal.prepare(replay_frame))
    failing_session = _AckSession(fail=True)
    with PostCommitObserverQueue(
        failing_session,
        ObserverRun(replay_frame.snapshot.provenance.run_identity),
        consumer_id="durable-replay",
    ) as queue:
        queue.submit(replay_frame, journal=journal, journal_record=pending)
        failed_reports = queue.flush()

    assert [report.status for report in failed_reports] == ["skipped"]
    assert [record.frame.identity for record in journal.list_pending()] == [replay_frame.identity]

    recovered = DurableJournal(root, sync="none", recover="automatic")
    (replay_record,) = recovered.list_pending()
    replay_session = _AckSession()
    with PostCommitObserverQueue(
        replay_session,
        ObserverRun(replay_frame.snapshot.provenance.run_identity),
        consumer_id="durable-replay",
    ) as queue:
        queue.submit(
            replay_record.frame,
            journal=recovered,
            journal_record=replay_record,
        )
        replay_reports = queue.flush()

    assert [report.status for report in replay_reports] == ["delivered"]
    assert [frame.identity for frame in replay_session.frames] == [replay_frame.identity]
    assert recovered.list_pending() == ()
    assert [record.state for record in recovered.list_committed()] == [
        "delivered",
        "delivered",
    ]


def test_prepared_live_frame_commits_only_after_transaction_seal(tmp_path: Path) -> None:
    journal = DurableJournal(tmp_path / "journal", sync="none", recover="manual")
    frame = _frame(field_name="sealed_temperature")
    record = journal.prepare(frame)
    calls = []
    effect = SimpleNamespace(
        identity=make_identity("accepted-side-effect", {"sample": 1}),
        payload=SimpleNamespace(identity=make_identity("consumer-payload", {"sample": 1})),
        target=SimpleNamespace(parallel_mode=ParallelMode.SERIAL),
    )

    def submit(effect_value, frame_value, journal_value, record_value, preexisting):
        assert preexisting is False
        committed = journal_value.commit(record_value)
        calls.append((effect_value, frame_value, journal_value, committed, preexisting))

    prepared = _PreparedLiveVisualization(
        effect,
        _detach_owned_observer_frame(frame),
        submit,
        journal,
        record,
    )

    prepared.publish()
    assert [item.state for item in journal.list_committed()] == []
    prepared.finalize()

    assert len(calls) == 1
    assert calls[0][0] is effect
    assert calls[0][2] is journal
    assert calls[0][3].state == "pending"
    assert [item.frame.identity for item in journal.list_pending()] == [frame.identity]


def test_per_rank_live_intent_authenticates_every_rank_handoff() -> None:
    effect = SimpleNamespace(
        identity=make_identity("accepted-side-effect", {"sample": "per-rank"}),
        payload=SimpleNamespace(
            identity=make_identity("consumer-payload", {"sample": "per-rank"})),
        target=SimpleNamespace(parallel_mode=ParallelMode.PER_RANK),
    )
    prepared = _PreparedLiveVisualization(
        effect,
        None,
        lambda *_args: None,
        size=2,
    )

    receipt = prepared.publish()

    assert receipt.parallel_mode is ParallelMode.PER_RANK
    assert tuple(rank for rank, _artifact in receipt.rank_artifacts) == (0, 1)
    assert len({artifact for _rank, artifact in receipt.rank_artifacts}) == 2


def test_runtime_replays_prior_run_pending_frame_from_scoped_journal(tmp_path: Path) -> None:
    configured = DurableJournal(tmp_path / "base", sync="none", recover="manual")
    frame = _frame(field_name="runtime_replay_temperature")
    run_identity = frame.snapshot.provenance.run_identity
    manifest = SimpleNamespace(
        qualified_id="monitor/durable-runtime",
        identity=make_identity("consumer-manifest", {"name": "durable-runtime"}),
        parallel_mode=ParallelMode.SERIAL,
        target_uri="durable-runtime",
        operation=SimpleNamespace(durability=configured),
    )
    publisher = RuntimeConsumerPublisher.__new__(RuntimeConsumerPublisher)
    publisher._rank = 0
    publisher._size = 1
    publisher._communicator = None
    publisher._observer_journals = {}
    publisher._owner = SimpleNamespace(_output_root=tmp_path / "outputs")
    journal = publisher._observer_journal(manifest, run_identity)
    pending = journal.commit(journal.prepare(frame))
    calls = []
    queue = SimpleNamespace(submit=lambda *args, **kwargs: calls.append((args, kwargs)))

    resumed_run = make_identity("run", {"name": "resumed-after-checkpoint"})
    assert resumed_run != run_identity
    records, states = publisher._inspect_observer_journal(manifest, journal)
    publisher._replay_observer_journal(manifest, queue, journal, records, states)

    assert len(calls) == 1
    assert calls[0][0] == (pending.frame,)
    assert calls[0][1] == {"journal": journal, "journal_record": pending}


def test_observer_run_explicitly_authorizes_prior_run_recovery() -> None:
    frame = _frame(field_name="recovered_temperature")
    previous = frame.snapshot.provenance.run_identity
    active = make_identity("run", {"name": "active-after-restart"})
    session = _AckSession()
    run = ObserverRun(active, recovery_run_identities=(previous,))

    with PostCommitObserverQueue(
        session, run, consumer_id="recovery-authority",
    ) as queue:
        queue.submit(frame)
        reports = queue.flush()

    assert [report.status for report in reports] == ["delivered"]
    assert session.run is not None
    assert session.run.accepted_run_identities == (active, previous)
