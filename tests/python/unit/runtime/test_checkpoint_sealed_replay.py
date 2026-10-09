"""Source-only I/O lifecycle tests: real NPZ files, no Native/PDE fixture or result."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import hashlib
import os
import fcntl

import numpy as np
import pytest

from pops.model import Handle, OwnerPath
from pops.output._consumer_contracts import (
    ConsumerCursorSet, ConsumerGraph, ConsumerKind, ConsumerManifest, ParallelMode, Retry,
    ScheduleCursor,
)
from pops.output._restart_provider import RestartV3, _CheckpointPayloadProof
from pops.runtime._checkpoint_resource_budget import _producer_checkpoint_resource_budget
from pops.runtime._consumer import ConsumerPublicationError, ConsumerPublisher, PreparedPublication, PublicationReceipt
from pops.runtime._consumer_transaction import ConsumerCursorAuthority, ConsumerTransaction
from pops.runtime._consumer_effects import AcceptedSideEffect, ConsumerPayload, EffectPlan, PublicationTarget
from pops.codegen.lowering_coverage import LoweringCoverageReport
from pops.identity import make_identity
from pops.runtime._runtime_consumers import _PreparedCheckpoint
from pops.time import AcceptedStep, Clock, Every, Schedule


class _ArchiveIO:
    """Only the provider seam; this NPZ is not a scientific/native checkpoint."""
    def __init__(self):
        self._execution_context = SimpleNamespace(communicator=SimpleNamespace(identity='serial', handle=None))
        self.payload = {'source_bytes': np.arange(64, dtype=np.float64)}
        self._checkpoint_resource_budget = _producer_checkpoint_resource_budget(
            self.payload, runtime_kind='uniform', authority='test-archive-io')
        self.captures = self.factories = self.validations = 0
        self.phase = 'prepared'
        self.candidate = object()
        self.original = b''

    def _prepare_checkpoint_candidate(self):
        self.factories += 1
        assert self.phase == 'prepared'
        return self.candidate

    def _validate_prepared_checkpoint_candidate(self, candidate):
        assert candidate is self.candidate and self.phase == 'prepared'

    def _validate_committed_checkpoint_candidate(self, candidate):
        self.validations += 1
        if candidate is not self.candidate or self.phase != 'committed':
            raise ValueError('checkpoint owner/session/committed point is stale')

    def _checkpoint_payload(self, path, *, transaction_receipt, prepared_capture):
        self.captures += 1
        assert prepared_capture is self.candidate and self.phase == 'prepared'
        entry = transaction_receipt.take_native_entry()
        with os.fdopen(entry.duplicate(), 'wb') as stream:
            np.savez_compressed(stream, **self.payload)
        self.original = Path(path).read_bytes()
        return _CheckpointPayloadProof(transaction_receipt, entry)


class _CheckpointPublisher(ConsumerPublisher):
    def __init__(self, runtime, target):
        self.runtime, self.target = runtime, target

    def prepare(self, effect):
        return _PreparedCheckpoint(effect, self.runtime, RestartV3(), self.target)


def _transaction_for(runtime, target, *, attempts=2, publisher=None):
    owner = OwnerPath.consumer('sealed-replay-source')
    clock = Clock('solution', owner=owner)
    manifest = ConsumerManifest(
        handle=Handle('checkpoint', kind='consumer', owner=owner),
        kind=ConsumerKind.CHECKPOINT, quantities=(),
        schedule=Schedule(Every(AcceptedStep(clock), 2)),
        target_uri=str(target), output_format=None, parallel_mode=ParallelMode.COLLECTIVE,
        operation=RestartV3(), failure_action=Retry(attempts),
    )
    # Exercise the real publication transaction with inert typed effect values. No compiled
    # artifact or ModelProgram is fabricated by this I/O unit fixture.
    bundle = make_identity('runtime-plan-bundle', dict(test='archive-io'))
    occurrence = make_identity('consumer-occurrence', dict(sample=1))
    before = ScheduleCursor(manifest.qualified_id)
    after = ScheduleCursor(manifest.qualified_id, occurrence.token, 1)
    payload = ConsumerPayload(bundle, occurrence, (), ())
    effect = AcceptedSideEffect(0, manifest.qualified_id, manifest.identity,
        PublicationTarget(str(target), None, manifest.operation_data, ParallelMode.COLLECTIVE),
        payload, Retry(attempts), before, after)
    plan = EffectPlan(ConsumerGraph((manifest,)).identity, bundle, (effect,), LoweringCoverageReport())
    cursors = ConsumerCursorSet()
    return ConsumerTransaction(plan, cursors, publisher or _CheckpointPublisher(runtime, target),
                               ConsumerCursorAuthority(cursors))


def _no_private_files(directory):
    assert list(directory.glob('.pops-restart-*')) == []


def test_retry_after_real_publication_uses_the_same_sealed_bytes_once(monkeypatch, tmp_path):
    runtime = _ArchiveIO()
    target = tmp_path/'accepted.npz'
    write = RestartV3.write
    images = []

    def fail_once(operation, snapshot, path):
        result = write(operation, snapshot, path)
        images.append(Path(result).read_bytes())
        if len(images) == 1:
            raise OSError('one real publication failure after link')
        return result

    monkeypatch.setattr(RestartV3, 'write', fail_once)
    transaction = _transaction_for(runtime, target)
    retained = transaction._prepared[0][1]._replay.proof.entry.fileno()
    assert fcntl.fcntl(retained, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
    runtime.phase = 'committed'
    report = transaction.accept()
    assert len(report.published) == 1
    assert report.published[0].publisher_id == 'pops.restart-checkpoint.v6'
    assert runtime.factories == runtime.captures == 1
    assert images == [runtime.original, runtime.original]
    assert hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(runtime.original).digest()
    assert target.stat().st_nlink == 2  # accepted target and retained source until seal
    assert transaction.seal() == ()
    assert target.stat().st_nlink == 1
    _no_private_files(tmp_path)


def test_exhaustion_compensates_all_attempts_and_releases_the_retained_inode(monkeypatch, tmp_path):
    runtime = _ArchiveIO()
    target = tmp_path/'accepted.npz'
    write = RestartV3.write
    publications = []

    def fail(operation, snapshot, path):
        result = write(operation, snapshot, path)
        publications.append(Path(result).read_bytes())
        raise OSError('publication budget exhausted')

    monkeypatch.setattr(RestartV3, 'write', fail)
    transaction = _transaction_for(runtime, target, attempts=3)
    runtime.phase = 'committed'
    with pytest.raises(ConsumerPublicationError):
        transaction.accept()
    assert publications == [runtime.original]*3
    assert runtime.factories == runtime.captures == 1
    assert not target.exists()
    _no_private_files(tmp_path)


def test_target_collision_never_replaces_foreign_bytes_and_cleans_replays(tmp_path):
    runtime = _ArchiveIO()
    target = tmp_path/'accepted.npz'
    target.write_bytes(b'existing accepted artifact')
    transaction = _transaction_for(runtime, target)
    runtime.phase = 'committed'
    with pytest.raises(ConsumerPublicationError):
        transaction.accept()
    assert target.read_bytes() == b'existing accepted artifact'
    assert runtime.factories == runtime.captures == 1
    _no_private_files(tmp_path)


def test_retention_refuses_an_archive_over_its_authenticated_live_budget(tmp_path):
    runtime = _ArchiveIO()
    runtime._checkpoint_resource_budget = replace(runtime._checkpoint_resource_budget, max_archive_bytes=1)
    with pytest.raises(ConsumerPublicationError):
        _transaction_for(runtime, tmp_path/'accepted.npz')
    _no_private_files(tmp_path)


def test_terminal_discard_releases_retention_without_a_publication(tmp_path):
    runtime = _ArchiveIO()
    transaction = _transaction_for(runtime, tmp_path/'accepted.npz')
    transaction.reject()
    assert runtime.captures == 1
    _no_private_files(tmp_path)


def test_replay_revalidates_the_committed_token_without_another_capture(monkeypatch, tmp_path):
    runtime = _ArchiveIO()
    target = tmp_path/'accepted.npz'
    write = RestartV3.write

    def invalidate(operation, snapshot, path):
        result = write(operation, snapshot, path)
        runtime.phase = 'finalized'
        raise OSError('publication failed and owner finalized')

    monkeypatch.setattr(RestartV3, 'write', invalidate)
    transaction = _transaction_for(runtime, target)
    runtime.phase = 'committed'
    with pytest.raises(ConsumerPublicationError, match='stale'):
        transaction.accept()
    assert runtime.factories == runtime.captures == 1
    assert not target.exists()
    _no_private_files(tmp_path)


def test_one_peer_staging_refusal_cleans_the_new_attempt_and_retained_link(monkeypatch, tmp_path):
    """Inject a control-plane peer vote; this is not a Native MPI execution."""
    import pops.output._restart_provider as provider
    import pops.output._checkpoint_collective as collective
    runtime = _ArchiveIO()
    target = tmp_path/'accepted.npz'
    write = RestartV3.write
    consensus = provider.consensus

    def fail(operation, snapshot, path):
        write(operation, snapshot, path)
        raise OSError('retry publication')

    def peer_refusal(topology, phase, **kwargs):
        if phase == 'sealed replay staging agreement':
            peer = collective._error_record(ValueError('peer refuses replay staging'))
            monkeypatch.setattr(collective, 'allgather_value', lambda comm, row: (
                dict(rank=0, value=None, error=None), dict(rank=1, value=None, error=peer)))
            return consensus(collective.CheckpointTopology(0, 2, object()), phase)
        return consensus(topology, phase, **kwargs)

    monkeypatch.setattr(RestartV3, 'write', fail)
    transaction = _transaction_for(runtime, target)
    runtime.phase = 'committed'
    monkeypatch.setattr(provider, 'consensus', peer_refusal)
    with pytest.raises(ConsumerPublicationError, match='peer refuses'):
        transaction.accept()
    assert runtime.captures == 1
    assert not target.exists()
    _no_private_files(tmp_path)


def test_release_failure_keeps_the_lease_for_idempotent_postcommit_finalize(monkeypatch, tmp_path):
    from pops.output._restart_provider import _SealedCheckpointReplay
    runtime = _ArchiveIO()
    target = tmp_path/'accepted.npz'
    transaction = _transaction_for(runtime, target)
    runtime.phase = 'committed'
    transaction.accept()
    cleanup = _SealedCheckpointReplay._cleanup_root
    calls = []

    def fail_once(replay):
        calls.append(1)
        if len(calls) == 1:
            raise OSError('retained source release unavailable')
        cleanup(replay)

    monkeypatch.setattr(_SealedCheckpointReplay, '_cleanup_root', fail_once)
    assert len(transaction.seal()) == 1
    assert target.exists() and target.stat().st_nlink == 2
    assert transaction.seal() == ()
    assert target.stat().st_nlink == 1
    _no_private_files(tmp_path)


def test_provider_without_replay_keeps_rollback_then_prepare_retry(tmp_path):
    events = []

    class LegacyPrepared(PreparedPublication):
        def __init__(self, effect):
            self.effect = effect
        @property
        def effect_identity(self):return self.effect.identity
        @property
        def payload_identity(self):return self.effect.payload.identity
        def publish(self):
            events.append('publish')
            if events.count('publish') == 1:
                raise OSError('legacy publication failure')
            return PublicationReceipt(self.effect_identity, self.payload_identity,
                                      'test-legacy-provider', 'test-artifact',
                                      self.effect.target.parallel_mode)
        def rollback(self):events.append('rollback')
        def discard(self):events.append('discard')

    class LegacyPublisher(ConsumerPublisher):
        def prepare(self, effect):
            events.append('prepare')
            return LegacyPrepared(effect)

    transaction = _transaction_for(_ArchiveIO(), tmp_path/'unused.npz', publisher=LegacyPublisher())
    transaction.accept()
    transaction.seal()
    assert events == ['prepare', 'publish', 'rollback', 'prepare', 'publish']


def test_refused_typed_replay_is_compensated_after_exclusive_lease_transfer(monkeypatch, tmp_path):
    runtime = _ArchiveIO()
    target = tmp_path/'accepted.npz'
    write, replay = RestartV3.write, _PreparedCheckpoint.retry_publication
    cleaned = []

    class WrongIdentity(PreparedPublication):
        def __init__(self, delegate):self.delegate = delegate
        @property
        def effect_identity(self):return make_identity('accepted-side-effect', dict(wrong=True))
        @property
        def payload_identity(self):return self.delegate.payload_identity
        def publish(self):raise AssertionError('refused replay must never publish')
        def discard(self):self.delegate.discard()
        def rollback(self):
            cleaned.append('rollback')
            self.delegate.rollback()

    def fail(operation, snapshot, path):
        write(operation, snapshot, path)
        raise OSError('first publication failed')

    def wrong(prepared, error):
        return WrongIdentity(replay(prepared, error))

    monkeypatch.setattr(RestartV3, 'write', fail)
    monkeypatch.setattr(_PreparedCheckpoint, 'retry_publication', wrong)
    transaction = _transaction_for(runtime, target, attempts=3)
    runtime.phase = 'committed'
    with pytest.raises(ConsumerPublicationError, match='exact effect payload'):
        transaction.accept()
    assert cleaned == ['rollback']
    assert runtime.factories == runtime.captures == 1
    assert not target.exists()
    _no_private_files(tmp_path)


def test_prospective_original_retry_case_uses_only_the_public_manifest_policy():
    import importlib.util
    import sys
    from tests.python.integration.runtime.test_imex_checkpoint_sealed_retry import EXAMPLE, _retry_target
    import pops
    spec = importlib.util.spec_from_file_location('original_retry_source_validation', EXAMPLE)
    example = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = example
    spec.loader.exec_module(example)
    target = _retry_target(example, output_mode=ParallelMode.SERIAL)
    pops.validate(target.authoring.case)
