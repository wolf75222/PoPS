"""Independent C34 membership review: real NPZ effects and nested publication roots."""

from dataclasses import replace

import pytest

from pops.codegen.lowering_coverage import LoweringCoverageReport
from pops.identity import make_identity
from pops.output import NPZ
from pops.output._consumer_contracts import (
    ConsumerCursorSet, FailRun, ParallelMode, ScheduleCursor,
)
from pops.runtime._consumer import (
    AcceptedSideEffect, ConsumerPayload, ConsumerTransaction, EffectPlan,
    PublicationTarget,
)
from pops.runtime._consumer_transaction import (
    ConsumerCursorAuthority, ConsumerPublisher, PreparedPublication,
)
from pops.runtime._output_publisher import ConsumerOutputPublisher, OutputPreparation
from tests.python.unit.output.test_exact_writers import _snapshot


class _BeforePublish(PreparedPublication):
    def __init__(self, inner, callback):
        self.inner, self.callback = inner, callback

    @property
    def effect_identity(self):
        return self.inner.effect_identity

    @property
    def payload_identity(self):
        return self.inner.payload_identity

    def publish(self):
        self.callback()
        return self.inner.publish()

    def discard(self):
        return self.inner.discard()

    def rollback(self):
        return self.inner.rollback()

    def finalize(self):
        return self.inner.finalize()


class _Publisher(ConsumerPublisher):
    def __init__(self, writer, callback):
        self.writer, self.callback = writer, callback

    def prepare(self, effect):
        prepared = self.writer.prepare(effect)
        return _BeforePublish(prepared, self.callback) if effect.ordinal == 0 else prepared


@pytest.mark.parametrize("nested_rollback", (False, True))
def test_mixed_membership_root_preserves_nested_and_later_disjoint_npz(tmp_path, nested_rollback):
    runtime = make_identity("runtime-plan-bundle", {"name": "independent-membership"})
    paths = {}

    def effect(consumer, sample, ordinal=0, before=None):
        occurrence = make_identity("consumer-occurrence", {"name": sample})
        path = tmp_path / (sample + ".npz")
        result = AcceptedSideEffect(
            ordinal, consumer, make_identity("consumer-manifest", {"name": consumer}),
            PublicationTarget(path.as_uri(), NPZ().consumer_data(), None, ParallelMode.SERIAL),
            ConsumerPayload(runtime, occurrence, (), ()), FailRun(),
            before or ScheduleCursor(consumer),
            ScheduleCursor(consumer, occurrence.token,
                           1 if before is None else before.committed_samples + 1),
        )
        paths[result.identity.token] = path
        return result

    absent = effect("absent", "outer-absent")
    explicit = effect("explicit", "outer-explicit", ordinal=1)
    nested = effect("nested", "nested-first")
    unrelated = ScheduleCursor("unrelated", "previous", 7)
    initial = ConsumerCursorSet((explicit.cursor_before, unrelated))
    authority = ConsumerCursorAuthority(initial)
    snapshot, request, _ = _snapshot()
    writer = ConsumerOutputPublisher(lambda item: OutputPreparation(
        NPZ(), snapshot, replace(request, consumer_id=item.consumer_id), paths[item.identity.token]))

    def transaction(items, publisher=writer):
        plan = EffectPlan(make_identity("consumer-graph", {"name": "independent-membership"}),
                          runtime, tuple(items), LoweringCoverageReport())
        return ConsumerTransaction(plan, authority.cursors, publisher, authority)

    inner = transaction((nested,))
    outer = transaction((absent, explicit), _Publisher(writer, inner.accept))
    outer.accept()  # Nested commit occurs after outer reservation, before its cursor commit.
    if nested_rollback:
        inner.rollback_accepted()
        later = effect("nested", "nested-retry")
    else:
        inner.seal()
        later = effect("nested", "nested-second", before=nested.cursor_after)
    last = transaction((later,))
    last.accept()
    last.seal()
    rejected = outer.rollback_accepted()

    expected = ConsumerCursorSet((*initial.rows, later.cursor_after))
    assert authority.cursors.to_data() == expected.to_data()
    assert rejected.cursors.to_data() == expected.to_data()
    assert not paths[absent.identity.token].exists()
    assert not paths[explicit.identity.token].exists()
    assert paths[nested.identity.token].exists() is (not nested_rollback)
    assert paths[later.identity.token].is_file()
    # Both outer reservations are released, including the one whose row disappeared.
    retry = transaction((absent, explicit))
    retry.accept()
    retry.rollback_accepted()
    assert authority.cursors.to_data() == expected.to_data()
