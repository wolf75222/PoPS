"""Checkpoint rank changes preserve all source mailboxes until native preparation."""

from types import SimpleNamespace

import numpy as np
import pytest

from pops.runtime._checkpoint_exchanges import prepare_checkpoint_continuation
from tests.python.unit.runtime.test_continuation_transitions import _owner


def _setup(monkeypatch, sources, *, rank=0, size=3):
    from pops.output import _checkpoint_collective

    owner = _owner()
    offsets = np.cumsum([0, *(len(source) for source in sources)], dtype=np.int64)
    payload = {
        "program_exchange_state": np.frombuffer(b"".join(sources), dtype=np.uint8).copy(),
        "program_exchange_offsets": offsets,
        "continuation_transition_plan": np.asarray(owner._continuation_transition_plan._json),
    }
    monkeypatch.setattr(_checkpoint_collective, "checkpoint_topology",
                        lambda unused: SimpleNamespace(rank=rank, size=size))
    return owner, payload


@pytest.mark.parametrize("identical", (False, True))
def test_rank_change_passes_every_source_to_pure_native_preparation(monkeypatch, identical):
    first = b"POPSEX02" + bytes(16)
    sources = (first, first if identical else first + b"another-shard")
    owner, payload = _setup(monkeypatch, sources, rank=2)
    original = owner._s.image
    calls = []

    def remap(images):
        assert type(images) is tuple
        assert all(type(image) is bytes for image in images)
        calls.append(images)
        return original

    owner._s._remap_checkpoint_program_exchanges = remap
    before = {name: value.copy() for name, value in payload.items()}
    assert prepare_checkpoint_continuation(owner, payload) == original
    assert calls == [sources]
    assert owner._s.validated == [original]
    assert owner._s.image == original
    for name, value in before.items():
        np.testing.assert_array_equal(payload[name], value)


def test_same_cardinality_selects_exact_target_rank_without_rematerialization(monkeypatch):
    sources = (b"POPSEX01" + bytes(8), b"POPSEX02" + bytes(8))
    owner, payload = _setup(monkeypatch, sources, rank=1, size=2)
    owner._s.image = sources[1]

    def forbidden(unused):
        pytest.fail("unchanged placement must retain the exact recorded rank image")

    owner._s._remap_checkpoint_program_exchanges = forbidden
    assert prepare_checkpoint_continuation(owner, payload) == sources[1]
    assert owner._s.validated == [sources[1]]


@pytest.mark.parametrize("result", (bytearray(b"POPSEX01"), b"x" * 4097))
def test_invalid_native_candidate_is_rejected_without_validation_or_publication(monkeypatch, result):
    owner, payload = _setup(monkeypatch, (b"POPSEX01" + bytes(8),))
    owner._s._remap_checkpoint_program_exchanges = lambda unused: result
    original = owner._s.image
    with pytest.raises(ValueError, match="rematerialized accepted exchanges"):
        prepare_checkpoint_continuation(owner, payload)
    assert owner._s.validated == []
    assert owner._s.image == original


def test_native_rejection_and_source_budget_leave_accepted_mailbox_unchanged(monkeypatch):
    owner, payload = _setup(monkeypatch, (b"POPSEX01" + bytes(8),))
    original = owner._s.image
    calls = []

    def reject(images):
        calls.append(images)
        raise ValueError("duplicate authenticated incidence")

    owner._s._remap_checkpoint_program_exchanges = reject
    with pytest.raises(ValueError, match="duplicate authenticated incidence"):
        prepare_checkpoint_continuation(owner, payload)
    assert len(calls) == 1
    owner._checkpoint_exchange_byte_capacity = 15
    with pytest.raises(ValueError, match="resolved byte capacity"):
        prepare_checkpoint_continuation(owner, payload)
    assert len(calls) == 1
    assert owner._s.validated == []
    assert owner._s.image == original


def test_rank_dependent_legacy_owner_cannot_fall_back_to_rank_zero(monkeypatch):
    owner, payload = _setup(monkeypatch, (b"POPSEX01" + bytes(8), b"POPSEX01" + bytes(16)))
    with pytest.raises(ValueError, match="cannot remap rank-dependent"):
        prepare_checkpoint_continuation(owner, payload)
    assert owner._s.validated == []
