"""Source protocol regression for diagnostic projection before checkpoint sealing.

The store below models revision/outer rollback only; these tests provide no Native
checkpoint or PDE evidence. Real ConsumerTransaction preparation and Retry publication
must reuse the same diagnostic values after the outer commit.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pops.identity import make_identity
from pops.model import Handle, OwnerPath
from pops.output._consumer_contracts import (
    ConsumerCursorSet, ConsumerKind, FailRun, Retry, SkipSampleReported,
)
from pops.output.data import DiagnosticKey, DiagnosticPayload
from pops.runtime._consumer import PreparedPublication, PublicationReceipt
from pops.runtime._consumer_transaction import ConsumerCursorAuthority
from pops.runtime._runtime_consumers import RuntimeConsumerPublisher, _diagnostic_record_name
from pops.runtime._runtime_instance import RuntimeInstance
from tests.python.unit.runtime.test_consumer_publication_c34 import _effect, _plan


class _DiagnosticStore:
    def __init__(self):
        self.values = {"old": 2.0}
        self.revision = 0
        self.committed = False
        self.events = []
        self._last_step_transaction_report = None

    def record_program_diagnostic(self, name, value):
        if self.committed:
            raise AssertionError("Native diagnostic recorder called after commit")
        self.events.append("record")
        self.values[name] = value
        self.revision += 1

    def _begin_step_transaction(self):
        self.before = dict(self.values), self.revision
        self.events.append("begin")

    def _commit_step_transaction(self):
        self.committed = True
        self.events.append("commit")

    def _finalize_step_transaction(self):
        self.events.append("finalize")

    def _rollback_step_transaction(self):
        self.values, self.revision = self.before
        self.committed = False
        self.events.append("rollback")

    def _prepare_checkpoint_candidate(self):
        self.events.append("candidate")
        return self.revision, dict(self.values)


class _Output(PreparedPublication):
    def __init__(self, effect, publisher):
        self.effect, self.publisher = effect, publisher

    @property
    def effect_identity(self):
        return self.effect.identity

    @property
    def payload_identity(self):
        return self.effect.payload.identity

    def publish(self):
        self.publisher.publish_attempts += 1
        self.publisher.store.events.append("output-publish")
        if self.publisher.fail_publication and self.publisher.publish_attempts == 1:
            raise OSError("scientific output publication failed")
        return PublicationReceipt(self.effect_identity, self.payload_identity, "source-output",
                                  make_identity("source-artifact", {}).token)

    def discard(self):
        pass

    def rollback(self):
        self.publisher.store.events.append("output-rollback")


class _CheckpointOperation:
    def __init__(self, publisher):
        self.publisher = publisher
        self.captures = []

    def snapshot(self, owner, directory, *, prepared_capture):
        assert owner is self.publisher._owner
        state = self.publisher.diagnostic_restart_state()
        self.captures.append((prepared_capture, state))
        return SimpleNamespace(discard=lambda: None, rollback=lambda: None,
                               capture=prepared_capture)

    def validate_snapshot(self, snapshot):
        assert snapshot.capture[0] == self.publisher.store.revision

    def write(self, snapshot, target):
        store = self.publisher.store
        assert snapshot.capture == (store.revision, store.values), "checkpoint candidate stale"
        store.events.append("checkpoint-publish")
        return target


class _Publisher(RuntimeConsumerPublisher):
    def __init__(self, owner, effects, *, fail_publication=False):
        self._owner, self.store = owner, owner._executor
        self._pending, self._pending_baselines = {}, {}
        self._diagnostics, self._baselines = {}, {}
        self._external_writers = {}
        self._rank, self._size, self._communicator = 0, 1, None
        self.fail_publication = fail_publication
        self.publish_attempts = self.evaluations = 0
        self._output = SimpleNamespace(prepare=lambda effect: _Output(effect, self))
        self.payload = DiagnosticPayload(
            DiagnosticKey(Handle("mass", kind="diagnostic", owner=OwnerPath.consumer("candidate")),
                          make_identity("component-manifest", {}), make_identity("layout", {}),
                          0, make_identity("diagnostic-state", {}).token, "integral"),
            3.25, "kg", {})
        self.checkpoint = _CheckpointOperation(self)
        self._by_id = {
            effect.consumer_id: SimpleNamespace(
                qualified_id=effect.consumer_id,
                identity=effect.manifest_identity,
                kind=ConsumerKind.CHECKPOINT if index else ConsumerKind.SCIENTIFIC_OUTPUT,
                diagnostic_quantities=() if index else (object(),),
                operation_data={"extension": ".npz"}, operation=self.checkpoint,
            ) for index, effect in enumerate(effects)
        }

    def _diagnostic_values(self, manifest):
        assert not self.store.committed, "diagnostic reduction repeated after commit"
        self.evaluations += 1
        return (self.payload,), {"baseline": 3.25}


class _Runtime(RuntimeInstance):
    def _moments(self, **kwargs):
        return (object(),)


def _runtime(monkeypatch, tmp_path, *, action=None, checkpoint=True):
    from pops.runtime import _runtime_instance as implementation
    output = replace(_effect("output", "sample-1", tmp_path / "out.npz"),
                     failure_action=FailRun() if action is None else action)
    effects = (output, replace(_effect("checkpoint", "checkpoint-sample-1", tmp_path / "restart.npz"),
                               ordinal=1))
    if not checkpoint:
        effects = effects[:1]
    plan = replace(_plan(output), effects=effects)
    runtime = _Runtime.__new__(_Runtime)
    runtime._executor = _DiagnosticStore()
    runtime._consumer_cursors = ConsumerCursorSet()
    runtime._consumer_cursor_authority = ConsumerCursorAuthority(runtime._consumer_cursors)
    runtime._consumer_reports = runtime._consumer_finalize_pending = ()
    runtime._consumer_recoveries = {}
    runtime._checkpoint_cursor_override = None
    runtime._attempt = 0
    runtime._output_root = tmp_path
    runtime._runtime_plan = object()
    runtime._publisher = _Publisher(runtime, effects)
    runtime._consumer_graph = SimpleNamespace(nodes=tuple(runtime._publisher._by_id.values()))
    monkeypatch.setattr(implementation, "plan_accepted_side_effects", lambda *args: plan)
    return runtime


@pytest.mark.parametrize("retry", [False, True], ids=["default", "scientific-output-retry"])
def test_output_publication_keeps_the_precommit_diagnostic_projection(monkeypatch, tmp_path, retry):
    runtime = _runtime(monkeypatch, tmp_path, action=Retry(2) if retry else None)
    publisher, store = runtime._publisher, runtime._executor
    publisher.fail_publication = retry
    runtime._accepted_step_transaction_body(lambda: ("accepted", 1))
    assert publisher.evaluations == 1
    assert store.events.count("record") == 1
    assert store.events.index("record") < store.events.index("candidate") < store.events.index("commit")
    assert store.events[-1] == "finalize"
    assert publisher.publish_attempts == (2 if retry else 1)
    assert len(publisher.checkpoint.captures) == 1
    capture, python_state = publisher.checkpoint.captures[0]
    assert capture == (store.revision, store.values)
    assert python_state["diagnostics"] == [publisher.payload.to_data()]
    assert python_state["baselines"] == {"baseline": (3.25).hex()}
    assert publisher.diagnostics == (publisher.payload,)
    assert publisher._pending == publisher._pending_baselines == {}
    assert runtime._checkpoint_cursor_override is None


def test_prepare_failure_restores_outer_diagnostics_and_clears_pending(monkeypatch, tmp_path):
    runtime = _runtime(monkeypatch, tmp_path)
    publisher, store = runtime._publisher, runtime._executor
    old_record = store.record_program_diagnostic

    def fail_after_record(name, value):
        old_record(name, value)
        raise RuntimeError("diagnostic record failed")

    store.record_program_diagnostic = fail_after_record
    with pytest.raises(Exception, match="diagnostic record failed"):
        runtime._accepted_step_transaction_body(lambda: (None, 1))
    assert store.values == {"old": 2.0} and store.revision == 0
    assert store.events[-1] == "rollback"
    assert publisher._pending == publisher._pending_baselines == {}
    assert publisher._diagnostics == publisher._baselines == {}
    assert runtime._checkpoint_cursor_override is None


def test_candidate_publisher_factory_failure_clears_predicted_cursor(monkeypatch, tmp_path):
    runtime = _runtime(monkeypatch, tmp_path)
    def fail_factory():
        assert runtime._checkpoint_cursor_override is not None
        raise RuntimeError("candidate publisher allocation failed")
    runtime._publisher.checkpoint_candidate_preparation = fail_factory
    with pytest.raises(RuntimeError, match="candidate publisher allocation failed"):
        runtime._stage_consumers(outer_rollback_authoritative=True)
    assert runtime._checkpoint_cursor_override is None


@pytest.mark.parametrize("at_start,at_end", [(True, False), (False, True)])
def test_initial_and_end_without_outer_keep_deferred_native_recording(monkeypatch, tmp_path, at_start, at_end):
    runtime = _runtime(monkeypatch, tmp_path)
    transactions = runtime._stage_consumers(at_start=at_start, at_end=at_end)
    assert runtime._executor.events == ["candidate"]
    assert runtime._publisher._pending
    for transaction in transactions:
        transaction.abort()
    assert runtime._publisher._pending == runtime._publisher._pending_baselines == {}


def test_skip_without_checkpoint_keeps_existing_diagnostic_publication(monkeypatch, tmp_path):
    runtime = _runtime(monkeypatch, tmp_path, action=SkipSampleReported(), checkpoint=False)
    transactions = runtime._stage_consumers(outer_rollback_authoritative=True)
    assert runtime._executor.events == []
    transactions[0].accept()
    assert runtime._executor.events == ["output-publish", "record"]
    transactions[0].seal()


def test_checkpoint_skip_keeps_existing_cursor_prediction_refusal(monkeypatch, tmp_path):
    runtime = _runtime(monkeypatch, tmp_path, action=SkipSampleReported())
    with pytest.raises(ValueError, match="may skip its sample"):
        runtime._stage_consumers(outer_rollback_authoritative=True)
    assert runtime._executor.events == []


@pytest.mark.parametrize("rank", [0, 1])
def test_one_rank_record_failure_blocks_every_writer_and_checkpoint(monkeypatch, tmp_path, rank):
    from pops.output import _checkpoint_collective

    runtime = _runtime(monkeypatch, tmp_path)
    publisher, store = runtime._publisher, runtime._executor
    publisher._rank, publisher._size, publisher._communicator = rank, 2, object()
    if rank == 1:
        def fail_record(name, value):
            raise RuntimeError("rank-one record failed")
        store.record_program_diagnostic = fail_record
    votes = []
    def gathered(communicator, envelope):
        assert communicator is publisher._communicator
        votes.append(envelope)
        error = _checkpoint_collective._error_record(RuntimeError("rank-one record failed"))
        return ({"rank": 0, "value": None, "error": None},
                {"rank": 1, "value": None, "error": error})
    monkeypatch.setattr(_checkpoint_collective, "allgather_value", gathered)
    with pytest.raises(Exception, match="rank-one record failed"):
        runtime._accepted_step_transaction_body(lambda: (None, 1))
    assert len(votes) == 1
    assert votes[0]["value"] is None
    assert "output-publish" not in store.events and "candidate" not in store.events
    assert store.events[-1] == "rollback"
    assert publisher._pending == publisher._pending_baselines == {}
    assert publisher._diagnostics == publisher._baselines == {}
    assert store.values == {"old": 2.0}
