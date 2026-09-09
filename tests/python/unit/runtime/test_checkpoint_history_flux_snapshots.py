from types import SimpleNamespace

import numpy as np
import pytest

from pops.runtime._checkpoint_history_flux_snapshots import (
    SNAPSHOT_OFFSETS_KEY,
    SNAPSHOT_STATE_KEY,
    capture_history_flux_snapshots,
    prepare_history_flux_snapshots,
)


def _consensus(topology, phase, *, error=None, value=None):
    if error is not None:
        raise error
    return tuple({"value": value} for _rank in range(topology.size))


def test_ranked_history_flux_snapshot_archive_roundtrips_and_preserves_empty_image(
    monkeypatch,
) -> None:
    from pops import _native_collectives
    from pops.output import _checkpoint_collective

    topology = SimpleNamespace(distributed=True, communicator=object(), size=2)
    rows = (b"POPSHFS1/rank/0", b"POPSHFS1/rank/1")
    monkeypatch.setattr(_native_collectives, "allgather_bytes", lambda communicator, local: rows)
    monkeypatch.setattr(
        _checkpoint_collective,
        "consensus",
        _consensus,
    )

    payload = {}
    canonical = b"POPSHFS1/canonical"

    def canonicalize(shards, source_ranks):
        assert shards == list(rows)
        assert source_ranks == 2
        return canonical

    capture_history_flux_snapshots(topology, rows[0], 64, canonicalize, payload)
    assert payload[SNAPSHOT_OFFSETS_KEY].tolist() == [0, len(canonical)]
    assert prepare_history_flux_snapshots(payload, shard_capacity=64) == (canonical,)

    empty = {}
    capture_history_flux_snapshots(
        topology,
        b"",
        0,
        lambda shards, source_ranks: pytest.fail("empty archive invoked its canonicalizer"),
        empty,
    )
    assert empty[SNAPSHOT_STATE_KEY].tolist() == []
    assert empty[SNAPSHOT_OFFSETS_KEY].tolist() == [0, 0]
    assert prepare_history_flux_snapshots(empty, shard_capacity=0) == (b"",)


def test_history_flux_snapshot_capture_fences_presence_and_local_capacity_before_gather(
    monkeypatch,
) -> None:
    from pops import _native_collectives
    from pops.output import _checkpoint_collective

    topology = SimpleNamespace(distributed=True, communicator=object(), size=2)
    gathers = []
    monkeypatch.setattr(
        _native_collectives,
        "allgather_bytes",
        lambda communicator, local: gathers.append(local),
    )

    def mixed_consensus(topology, phase, *, error=None, value=None):
        if error is not None:
            raise error
        return ({"value": value}, {"value": {**value, "present": not value["present"]}})

    monkeypatch.setattr(_checkpoint_collective, "consensus", mixed_consensus)
    with pytest.raises(RuntimeError, match="plans differ"):
        capture_history_flux_snapshots(
            topology, b"POPSHFS1/rank/0", 64, lambda shards, source_ranks: b"", {}
        )
    assert gathers == []

    monkeypatch.setattr(_checkpoint_collective, "consensus", _consensus)
    with pytest.raises(ValueError, match="artifact capacity"):
        capture_history_flux_snapshots(
            topology, b"oversized", 1, lambda shards, source_ranks: b"", {}
        )
    assert gathers == []


def test_history_flux_snapshot_capture_fences_canonicalization_and_agrees_digest(
    monkeypatch,
) -> None:
    from pops import _native_collectives
    from pops.output import _checkpoint_collective

    topology = SimpleNamespace(distributed=True, communicator=object(), size=2)
    rows = (b"POPSHFS1/rank/0", b"POPSHFS1/rank/1")
    monkeypatch.setattr(_native_collectives, "allgather_bytes", lambda communicator, local: rows)
    calls = []

    def disagreeing_consensus(topology, phase, *, error=None, value=None):
        calls.append(phase)
        if error is not None:
            raise error
        if phase.endswith("canonicalization"):
            return ({"value": value}, {"value": "0" * 64})
        return _consensus(topology, phase, value=value)

    monkeypatch.setattr(_checkpoint_collective, "consensus", disagreeing_consensus)
    payload = {}
    with pytest.raises(RuntimeError, match="differs across ranks"):
        capture_history_flux_snapshots(
            topology, rows[0], 64, lambda shards, source_ranks: b"canonical", payload
        )
    assert calls == [
        "history-flux snapshot checkpoint preflight",
        "history-flux snapshot checkpoint canonicalization",
    ]
    assert payload == {}

    monkeypatch.setattr(_checkpoint_collective, "consensus", _consensus)

    def refuse_canonicalization(shards, source_ranks):
        raise ValueError("canonicalization refused")

    with pytest.raises(ValueError, match="canonicalization refused"):
        capture_history_flux_snapshots(
            topology,
            rows[0],
            64,
            refuse_canonicalization,
            {},
        )


@pytest.mark.parametrize(
    ("state", "offsets", "capacity", "message"),
    (
        (np.array([1], dtype=np.int16), np.array([0, 1], dtype=np.int64), 64, "uint8"),
        (np.array([1], dtype=np.uint8), np.array([0, 1], dtype=np.int32), 64, "int64"),
        (np.array([1], dtype=np.uint8), np.array([0, 0], dtype=np.int64), 64, "canonical"),
        (
            np.array([1, 2], dtype=np.uint8),
            np.array([0, 2, 2], dtype=np.int64),
            64,
            "canonical",
        ),
        (np.array([1, 2], dtype=np.uint8), np.array([0, 2], dtype=np.int64), 1, "capacity"),
    ),
)
def test_history_flux_snapshot_preflight_refuses_malformed_rank_archive(
    state, offsets, capacity, message
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_history_flux_snapshots(
            {
                SNAPSHOT_STATE_KEY: state,
                SNAPSHOT_OFFSETS_KEY: offsets,
            },
            shard_capacity=capacity,
        )


def test_history_flux_snapshot_preflight_preserves_legacy_absence_but_refuses_partial_pair() -> (
    None
):
    assert prepare_history_flux_snapshots({}, shard_capacity=64) is None
    with pytest.raises(ValueError, match="partial"):
        prepare_history_flux_snapshots(
            {SNAPSHOT_STATE_KEY: np.array([], dtype=np.uint8)},
            shard_capacity=64,
        )
