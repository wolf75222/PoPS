"""C34 protocol witness with exact carriers and real NPZ publications."""

from dataclasses import replace

import pytest

from pops.codegen.lowering_coverage import LoweringCoverageReport
from pops.identity import make_identity
from pops.output import NPZ
from pops.output._consumer_contracts import ConsumerCursorSet, ScheduleCursor
from pops.runtime._consumer import (
    AcceptedSideEffect, ConsumerPayload, ConsumerTransaction, EffectPlan,
    PublicationTarget, ConsumerPublicationError,
)
from pops.runtime._consumer_transaction import ConsumerCursorAuthority
from pops.output._consumer_contracts import FailRun, ParallelMode
from pops.runtime._output_publisher import ConsumerOutputPublisher, OutputPreparation
from tests.python.unit.output.test_exact_writers import _snapshot


def _effect(consumer_id, occurrence, target):
    occurrence_id = make_identity("consumer-occurrence", {"name": occurrence})
    payload = ConsumerPayload(
        make_identity("runtime-plan-bundle", {"name": "c34"}), occurrence_id, (), ())
    before = ScheduleCursor(consumer_id)
    after = ScheduleCursor(consumer_id, occurrence_id.token, 1)
    return AcceptedSideEffect(
        0, consumer_id,
        make_identity("consumer-manifest", {"name": consumer_id}),
        PublicationTarget(target.as_uri(), NPZ().consumer_data(), None, ParallelMode.SERIAL),
        payload, FailRun(), before, after)


def _plan(effect):
    return EffectPlan(
        make_identity("consumer-graph", {"name": "c34"}),
        effect.payload.runtime_plan_identity,
        (effect,), LoweringCoverageReport())


def _publisher(effects):
    snapshot, request, _ = _snapshot()
    def resolve(effect):
        return OutputPreparation(
            NPZ(), snapshot, replace(request, consumer_id=effect.consumer_id),
            effects[effect.identity.token])
    return ConsumerOutputPublisher(resolve)


def test_disjoint_roots_keep_both_artifacts_and_cursors(tmp_path):
    left_path, right_path = tmp_path / "left.npz", tmp_path / "right.npz"
    left = _effect("left", "left-2", left_path)
    right = _effect("right", "right-2", right_path)
    initial = ConsumerCursorSet()
    authority = ConsumerCursorAuthority(initial)
    publisher = _publisher({left.identity.token: left_path, right.identity.token: right_path})
    transactions = (
        ConsumerTransaction(_plan(left), initial, publisher, authority),
        ConsumerTransaction(_plan(right), initial, publisher, authority),
    )
    transactions[0].accept()
    transactions[1].accept()
    transactions[0].seal()
    transactions[1].seal()
    cursors = authority.cursors
    assert left_path.is_file() and right_path.is_file()
    assert cursors.for_consumer("left").committed_samples == 1
    assert cursors.for_consumer("right").committed_samples == 1


def test_conflicting_roots_refuse_before_second_publication(tmp_path):
    first_path, second_path = tmp_path / "first.npz", tmp_path / "second.npz"
    first = _effect("shared", "shared-2", first_path)
    second = _effect("shared", "shared-4", second_path)
    initial = ConsumerCursorSet()
    authority = ConsumerCursorAuthority(initial)
    publisher = _publisher({first.identity.token: first_path, second.identity.token: second_path})
    transactions = (
        ConsumerTransaction(_plan(first), initial, publisher, authority),
        ConsumerTransaction(_plan(second), initial, publisher, authority),
    )
    transactions[0].accept()
    with pytest.raises(ConsumerPublicationError, match="reserved"):
        transactions[1].accept()
    assert first_path.is_file() and not second_path.exists()
    assert authority.cursors.for_consumer("shared").committed_samples == 1
    transactions[0].seal()


def test_stale_plan_rejected_at_construction(tmp_path):
    path = tmp_path / "one.npz"
    effect = _effect("one", "one-2", path)
    initial = ConsumerCursorSet()
    authority = ConsumerCursorAuthority(initial)
    publisher = _publisher({effect.identity.token: path})
    transaction = ConsumerTransaction(_plan(effect), initial, publisher, authority)
    accepted = transaction.accept()
    with pytest.raises(ValueError, match="cursor snapshot is stale"):
        ConsumerTransaction(_plan(effect), accepted.cursors, publisher, authority)
    transaction.seal()


def test_prepared_stale_root_refuses_before_publication_after_other_seals(tmp_path):
    first_path, stale_path = tmp_path / "first.npz", tmp_path / "stale.npz"
    first = _effect("shared", "first", first_path)
    stale = _effect("shared", "stale", stale_path)
    initial = ConsumerCursorSet()
    authority = ConsumerCursorAuthority(initial)
    publisher = _publisher({first.identity.token: first_path, stale.identity.token: stale_path})
    first_tx = ConsumerTransaction(_plan(first), initial, publisher, authority)
    stale_tx = ConsumerTransaction(_plan(stale), initial, publisher, authority)
    first_tx.accept()
    first_tx.seal()
    with pytest.raises(ConsumerPublicationError, match="stale"):
        stale_tx.accept()
    assert first_path.is_file() and not stale_path.exists()
    assert authority.cursors.for_consumer("shared") == first.cursor_after


def test_disjoint_rollback_does_not_restore_over_other_root(tmp_path):
    left_path, right_path = tmp_path / "left.npz", tmp_path / "right.npz"
    left, right = _effect("left", "left", left_path), _effect("right", "right", right_path)
    initial = ConsumerCursorSet()
    authority = ConsumerCursorAuthority(initial)
    publisher = _publisher({left.identity.token: left_path, right.identity.token: right_path})
    left_tx = ConsumerTransaction(_plan(left), initial, publisher, authority)
    right_tx = ConsumerTransaction(_plan(right), initial, publisher, authority)
    left_tx.accept()
    right_tx.accept()
    right_tx.seal()
    left_tx.rollback_accepted()
    assert not left_path.exists() and right_path.is_file()
    assert authority.cursors.for_consumer("left") == left.cursor_before
    assert authority.cursors.for_consumer("right") == right.cursor_after


def test_failed_publication_releases_cursor_for_real_retry(tmp_path):
    target = tmp_path / "blocked.npz"
    target.mkdir()
    effect = _effect("retry", "retry", target)
    initial = ConsumerCursorSet()
    authority = ConsumerCursorAuthority(initial)
    publisher = _publisher({effect.identity.token: target})
    first = ConsumerTransaction(_plan(effect), initial, publisher, authority)
    with pytest.raises(ConsumerPublicationError):
        first.accept()
    assert authority.cursors.for_consumer("retry") == effect.cursor_before
    target.rmdir()
    second = ConsumerTransaction(_plan(effect), initial, publisher, authority)
    second.accept()
    second.seal()
    assert target.is_file()
    assert authority.cursors.for_consumer("retry") == effect.cursor_after
