"""Independent real-output tests for shared consumer cursor authority.

These qualify publication and cursor isolation, not concurrent PDE execution.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from types import SimpleNamespace

import pytest

from pops.output._consumer_contracts import ConsumerCursorSet
from pops.runtime._consumer_transaction import (
    ConsumerCursorAuthority, ConsumerPublicationError, ConsumerPublisher, ConsumerTransaction,
    PreparedPublication,
)
from tests.python.unit.runtime.test_consumer_publication_c34 import _effect, _plan, _publisher


def _transactions(tmp_path, consumers):
    paths = tuple(tmp_path / ("root%d.npz" % i) for i in range(len(consumers)))
    effects = tuple(_effect(name, "occurrence%d" % i, path)
                    for i, (name, path) in enumerate(zip(consumers, paths, strict=True)))
    initial = ConsumerCursorSet()
    authority = ConsumerCursorAuthority(initial)
    publisher = _publisher({effect.identity.token: path
                            for effect, path in zip(effects, paths, strict=True)})
    transactions = tuple(ConsumerTransaction(_plan(effect), initial, publisher, authority)
                         for effect in effects)
    return paths, effects, authority, publisher, transactions


def test_compensation_of_one_root_preserves_interleaved_disjoint_publication(tmp_path):
    paths, _, authority, _, (first, second) = _transactions(tmp_path, ("first", "second"))
    first.accept()
    second.accept()
    second.seal()
    retained_bytes = paths[1].read_bytes()
    first.rollback_accepted()
    assert not paths[0].exists()
    assert paths[1].read_bytes() == retained_bytes
    assert authority.cursors.for_consumer("first").committed_samples == 0
    assert authority.cursors.for_consumer("second").committed_samples == 1


@pytest.mark.parametrize("seal_first", (False, True))
def test_conflicting_root_rejected_before_its_npz_is_visible(tmp_path, seal_first):
    paths, _, authority, _, (first, second) = _transactions(tmp_path, ("shared", "shared"))
    first.accept()
    retained_bytes = paths[0].read_bytes()
    if seal_first:
        first.seal()
    with pytest.raises(ConsumerPublicationError, match="reserved|stale"):
        second.accept()
    assert paths[0].read_bytes() == retained_bytes and not paths[1].exists()
    assert authority.cursors.for_consumer("shared").committed_samples == 1
    if not seal_first:
        first.seal()


def test_checkpoint_reset_revokes_prepared_roots_even_when_cursors_are_identical(tmp_path):
    paths, effects, authority, publisher, (old,) = _transactions(tmp_path, ("shared",))
    # A cursor restored to the same value is not authority for an old payload.
    authority.reset(ConsumerCursorSet())
    with pytest.raises(ConsumerPublicationError, match="incarnation|generation|epoch|stale|reset"):
        old.accept()
    assert not paths[0].exists()
    fresh = ConsumerTransaction(_plan(effects[0]), authority.cursors, publisher, authority)
    fresh.accept()
    fresh.seal()
    assert paths[0].is_file()
    assert authority.cursors.for_consumer("shared").committed_samples == 1


def test_compensated_root_allows_fresh_attempt_without_cursor_double_count(tmp_path):
    paths, effects, authority, publisher, (old,) = _transactions(tmp_path, ("shared",))
    old.accept()
    old.rollback_accepted()
    assert not paths[0].exists()
    fresh = ConsumerTransaction(_plan(effects[0]), authority.cursors, publisher, authority)
    fresh.accept()
    fresh.seal()
    assert paths[0].is_file()
    assert authority.cursors.for_consumer("shared").committed_samples == 1


def test_productive_transaction_cannot_implicitly_create_private_authority(tmp_path):
    effect = _effect("shared", "old", tmp_path / "unauthorized.npz")
    publisher = _publisher({effect.identity.token: tmp_path / "unauthorized.npz"})
    with pytest.raises(TypeError, match="authority"):
        ConsumerTransaction(_plan(effect), ConsumerCursorSet(), publisher)
    assert not (tmp_path / "unauthorized.npz").exists()


def test_simultaneous_conflicting_roots_have_one_visible_winner(tmp_path):
    paths, _, authority, _, transactions = _transactions(tmp_path, ("shared", "shared"))
    barrier = Barrier(2)

    def accept(transaction):
        barrier.wait(timeout=5)
        try:
            transaction.accept()
            return True
        except ConsumerPublicationError:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        winners = tuple(executor.map(accept, transactions))
    assert sum(winners) == 1
    assert tuple(path.exists() for path in paths) == winners
    assert authority.cursors.for_consumer("shared").committed_samples == 1
    transactions[winners.index(True)].seal()


class _FailAfterPublish(PreparedPublication):
    """Inject failure after real NPZ publication, preserving actual rollback."""
    def __init__(self, prepared):
        self.prepared = prepared

    @property
    def effect_identity(self):
        return self.prepared.effect_identity

    @property
    def payload_identity(self):
        return self.prepared.payload_identity

    def publish(self):
        self.prepared.publish()
        raise OSError("injected after real NPZ publication")

    def discard(self):
        return self.prepared.discard()

    def rollback(self):
        return self.prepared.rollback()


class _FailSecondPublisher(ConsumerPublisher):
    def __init__(self, delegate, consumer_id):
        self.delegate, self.consumer_id = delegate, consumer_id

    def prepare(self, effect):
        prepared = self.delegate.prepare(effect)
        return (_FailAfterPublish(prepared) if effect.consumer_id == self.consumer_id else prepared)


def test_partial_publication_failure_releases_whole_root_and_preserves_other_root(tmp_path):
    paths = tuple(tmp_path / (name + ".npz") for name in ("first", "second", "disjoint"))
    effects = tuple(_effect(name, name + "-one", path)
                    for name, path in zip(("first", "second", "disjoint"), paths, strict=True))
    effects = (effects[0], replace(effects[1], ordinal=1), effects[2])
    initial = ConsumerCursorSet()
    authority = ConsumerCursorAuthority(initial)
    publisher = _publisher({effect.identity.token: path
                            for effect, path in zip(effects, paths, strict=True)})
    root_plan = replace(_plan(effects[0]), effects=effects[:2])
    root = ConsumerTransaction(root_plan, initial, _FailSecondPublisher(publisher, "second"), authority)
    other = ConsumerTransaction(_plan(effects[2]), initial, publisher, authority)
    other.accept()
    other.seal()
    retained_bytes = paths[2].read_bytes()
    with pytest.raises(ConsumerPublicationError, match="after real NPZ"):
        root.accept()
    assert not paths[0].exists() and not paths[1].exists()
    assert paths[2].read_bytes() == retained_bytes
    assert authority.cursors.for_consumer("first").committed_samples == 0
    assert authority.cursors.for_consumer("second").committed_samples == 0
    assert authority.cursors.for_consumer("disjoint").committed_samples == 1
    retry = ConsumerTransaction(root_plan, authority.cursors, publisher, authority)
    retry.accept()
    retry.seal()
    assert all(path.exists() for path in paths)
    assert all(authority.cursors.for_consumer(name).committed_samples == 1
               for name in ("first", "second", "disjoint"))


def test_controller_envelope_rollback_preserves_a_sealed_disjoint_root(tmp_path):
    from pops.runtime._runtime_instance import RuntimeInstance

    paths, _, authority, _, (first, second) = _transactions(tmp_path, ("first", "second"))
    # Exercise the real controller envelope, with no PDE/native-executor claim.
    runtime = object.__new__(RuntimeInstance)
    runtime._executor = SimpleNamespace()
    runtime._attempt = 0
    runtime._consumer_cursors = authority.cursors
    runtime._consumer_cursor_authority = authority
    runtime._consumer_reports = ()
    runtime._checkpoint_cursor_override = None
    snapshot = runtime._step_envelope_snapshot()
    first.accept()
    second_report = second.accept()
    second.seal()
    runtime._consumer_reports = (second_report,)
    first.rollback_accepted()
    retained_bytes = paths[1].read_bytes()
    runtime._restore_step_envelope(snapshot)
    assert not paths[0].exists() and paths[1].read_bytes() == retained_bytes
    assert authority.cursors.for_consumer("first").committed_samples == 0
    assert authority.cursors.for_consumer("second").committed_samples == 1
    assert runtime._consumer_cursors == authority.cursors
    assert runtime._consumer_reports == (second_report,)


def test_incomplete_compensation_keeps_conflicting_roots_and_reset_blocked(tmp_path):
    class Uncompensated(_FailAfterPublish):
        def rollback(self):
            raise OSError("injected incomplete compensation")

    class Publisher(ConsumerPublisher):
        def prepare(self, effect):
            return Uncompensated(publisher.prepare(effect))

    first_path, second_path = tmp_path / "incomplete.npz", tmp_path / "retry.npz"
    first = _effect("shared", "one", first_path)
    second = _effect("shared", "two", second_path)
    initial = ConsumerCursorSet()
    authority = ConsumerCursorAuthority(initial)
    publisher = _publisher({first.identity.token: first_path, second.identity.token: second_path})
    failed = ConsumerTransaction(_plan(first), initial, Publisher(), authority)
    with pytest.raises(ConsumerPublicationError):
        failed.accept()
    assert first_path.exists()
    assert authority.cursors.for_consumer("shared").committed_samples == 0
    fresh = ConsumerTransaction(_plan(second), initial, publisher, authority)
    with pytest.raises(ConsumerPublicationError, match="incomplete.*compensation.*recreate RuntimeInstance"):
        fresh.accept()
    assert not second_path.exists()
    with pytest.raises(RuntimeError, match="incomplete.*compensation.*recreate RuntimeInstance"):
        authority.reset(initial)
    other_path = tmp_path / "unaffected-consumer.npz"
    other_effect = _effect("unaffected", "one", other_path)
    other_publisher = _publisher({other_effect.identity.token: other_path})
    other = ConsumerTransaction(_plan(other_effect), authority.cursors, other_publisher, authority)
    other.accept()
    other.seal()
    assert other_path.is_file()
    assert authority.cursors.for_consumer("unaffected").committed_samples == 1
    assert authority.cursors.for_consumer("shared").committed_samples == 0


def test_checkpoint_cursor_cas_refuses_to_erase_a_new_sealed_publication(tmp_path):
    paths, _, authority, _, (publication,) = _transactions(tmp_path, ("other-root",))
    old_cursors, old_revision = authority.snapshot()
    publication.accept()
    publication.seal()
    retained_bytes = paths[0].read_bytes()
    with pytest.raises(RuntimeError, match="revision|changed|stale"):
        authority.reset(old_cursors, expected_revision=old_revision)
    assert paths[0].read_bytes() == retained_bytes
    assert authority.cursors.for_consumer("other-root").committed_samples == 1


def test_checkpoint_rollback_cas_preserves_progress_after_its_own_install(tmp_path):
    paths, effects, authority, publisher, (obsolete,) = _transactions(tmp_path, ("after-install",))
    before_install, original_revision = authority.snapshot()
    installed_revision = authority.reset(before_install, expected_revision=original_revision)
    assert type(installed_revision) is int and installed_revision > original_revision
    obsolete.reject()
    fresh = ConsumerTransaction(_plan(effects[0]), authority.cursors, publisher, authority)
    fresh.accept()
    fresh.seal()
    retained_bytes = paths[0].read_bytes()
    with pytest.raises(RuntimeError, match="revision|changed|stale"):
        authority.reset(before_install, expected_revision=installed_revision)
    assert paths[0].read_bytes() == retained_bytes
    assert authority.cursors.for_consumer("after-install").committed_samples == 1
